from __future__ import annotations

import os
from pathlib import Path
from datetime import datetime, date
from typing import List, Tuple

import pandas as pd


def parse_report_date_from_filename(filename: str) -> date | None:
    """
    Parse date from a JHU US daily report filename in the format MM-DD-YYYY.csv.

    Returns a datetime.date or None if the filename doesn't match the pattern.
    """
    stem = Path(filename).stem
    try:
        return datetime.strptime(stem, "%m-%d-%Y").date()
    except Exception:
        return None


def list_daily_files_in_range(daily_dir: Path, start_date: date, end_date: date) -> List[Tuple[date, Path]]:
    """
    List all CSV files in the given directory whose filename-encoded date falls within [start_date, end_date].
    Returns a list of (report_date, file_path) tuples, sorted by report_date.
    """
    candidates: List[Tuple[date, Path]] = []
    for entry in daily_dir.iterdir():
        if not entry.is_file():
            continue
        if entry.suffix.lower() != ".csv":
            continue
        report_date = parse_report_date_from_filename(entry.name)
        if report_date is None:
            continue
        if start_date <= report_date <= end_date:
            candidates.append((report_date, entry))
    candidates.sort(key=lambda x: x[0])
    return candidates


def read_daily_report(file_path: Path, report_date: date) -> pd.DataFrame:
    """
    Read a single JHU daily US report CSV as DataFrame and attach a `Report_Date` column.
    Ensures certain identifier columns stay as string to avoid float coercion issues.
    """
    # Read without forcing parse_dates (some early files may lack certain columns)
    df = pd.read_csv(
        file_path,
        dtype={
            "FIPS": "string",
            "UID": "string",
            "ISO3": "string",
            "Province_State": "string",
            "Country_Region": "string",
        },
        encoding="utf-8-sig",
    )

    # Attach normalized report date derived from filename
    df["Report_Date"] = pd.to_datetime(report_date)

    # Coerce potential numeric columns to numeric (softly)
    numeric_like_columns = [
        "Lat",
        "Long_",
        "Confirmed",
        "Deaths",
        "Recovered",
        "Active",
        "Incident_Rate",
        "Total_Test_Results",
        "People_Hospitalized",
        "Case_Fatality_Ratio",
        "Testing_Rate",
        "Hospitalization_Rate",
        "People_Tested",
        "Mortality_Rate",
    ]
    for col in numeric_like_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Parse optional date-time columns if present
    if "Last_Update" in df.columns:
        df["Last_Update"] = pd.to_datetime(df["Last_Update"], errors="coerce")
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    return df


def filter_states(df: pd.DataFrame, include_states: List[str]) -> pd.DataFrame:
    """
    Filter DataFrame to include only specified states, excluding cruise ships.
    """
    exclude = {"Diamond Princess", "Grand Princess"}
    mask_valid = ~df["Province_State"].isin(exclude)
    mask_state = df["Province_State"].isin(include_states)
    return df.loc[mask_valid & mask_state].copy()


def aggregate_us(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate all rows (excluding cruise ships) for a single day's DataFrame into a national-level row.

    - Sums additive metrics (Confirmed, Deaths, Recovered, Active, tests, hospitalized, etc.).
    - Recomputes Case_Fatality_Ratio = Deaths / Confirmed * 100 when possible.
    - Leaves non-additive rate fields (Incident_Rate, Testing_Rate, etc.) as NaN.
    Returns a one-row DataFrame.
    """
    exclude = {"Diamond Princess", "Grand Princess"}
    df = df.loc[~df["Province_State"].isin(exclude)].copy()

    # Identify additive numeric columns to sum
    additive_columns = [
        "Confirmed",
        "Deaths",
        "Recovered",
        "Active",
        "Total_Test_Results",
        "People_Hospitalized",
        "People_Tested",
    ]
    for col in additive_columns:
        if col not in df.columns:
            df[col] = pd.NA
        else:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    sums = {col: df[col].sum(min_count=1) for col in additive_columns}

    report_date_value = df["Report_Date"].iloc[0] if "Report_Date" in df.columns and not df.empty else pd.NaT

    # Compute CFR if possible
    confirmed_sum = sums.get("Confirmed")
    deaths_sum = sums.get("Deaths")
    cfr_value = (float(deaths_sum) / float(confirmed_sum) * 100.0) if pd.notna(confirmed_sum) and confirmed_sum and pd.notna(deaths_sum) else pd.NA

    # Build one-row dataframe with canonical columns we can provide
    data = {
        "Province_State": "US",
        "Country_Region": "US",
        "Report_Date": report_date_value,
        **sums,
        "Case_Fatality_Ratio": cfr_value,
    }
    us_row = pd.DataFrame([data])
    return us_row


def main() -> None:
    # Configuration
    project_root = Path(__file__).parent
    daily_dir = project_root / "COVID-19" / "csse_covid_19_data" / "csse_covid_19_daily_reports_us"
    output_dir = project_root / "data"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Date range inclusive
    start_date = date(2020, 4, 10)
    end_date = date(2020, 6, 20)

    # Target states
    target_states = ["New York", "Texas"]

    # Collect per-day dataframes
    files_in_range = list_daily_files_in_range(daily_dir, start_date, end_date)
    if not files_in_range:
        raise FileNotFoundError(f"No daily report files found in range {start_date} to {end_date} under {daily_dir}")

    ny_tx_frames: List[pd.DataFrame] = []
    us_agg_frames: List[pd.DataFrame] = []

    for report_date, file_path in files_in_range:
        df_day = read_daily_report(file_path, report_date)

        # Filter NY/TX
        df_ny_tx = filter_states(df_day, target_states)
        ny_tx_frames.append(df_ny_tx)

        # US aggregate for that day
        us_row = aggregate_us(df_day)
        us_agg_frames.append(us_row)

    # Concatenate over time
    ny_tx_all = pd.concat(ny_tx_frames, ignore_index=True, sort=False)
    us_agg_all = pd.concat(us_agg_frames, ignore_index=True, sort=False)

    # Sort by state then date for readability
    if "Report_Date" in ny_tx_all.columns:
        ny_tx_all = ny_tx_all.sort_values(by=["Province_State", "Report_Date"])  # type: ignore[arg-type]
    if "Report_Date" in us_agg_all.columns:
        us_agg_all = us_agg_all.sort_values(by=["Report_Date"])  # type: ignore[arg-type]

    # Write outputs
    ny_tx_path = output_dir / "ny_tx_2020-04-10_to_2020-06-20.csv"
    us_path = output_dir / "us_aggregate_2020-04-10_to_2020-06-20.csv"

    ny_tx_all.to_csv(ny_tx_path, index=False)
    us_agg_all.to_csv(us_path, index=False)

    print(f"Saved NY/TX filtered dataset to: {ny_tx_path}")
    print(f"Saved US aggregate dataset to: {us_path}")

    # Also write slimmed versions with only LLM-relevant columns
    keep_cols_common = [
        "Province_State",
        "Report_Date",
        "Confirmed",
        "Deaths",
        "Active",
        "Case_Fatality_Ratio",
        "Total_Test_Results",
        "People_Tested",
        "People_Hospitalized",
    ]

    def slim(df: pd.DataFrame, keep_cols: list[str]) -> pd.DataFrame:
        existing = [c for c in keep_cols if c in df.columns]
        return df.loc[:, existing].copy()

    ny_tx_slim = slim(ny_tx_all, keep_cols_common)
    us_agg_slim = slim(us_agg_all, keep_cols_common)

    ny_tx_slim_path = output_dir / "ny_tx_2020-04-10_to_2020-06-20_slim.csv"
    us_slim_path = output_dir / "us_aggregate_2020-04-10_to_2020-06-20_slim.csv"

    ny_tx_slim.to_csv(ny_tx_slim_path, index=False)
    us_agg_slim.to_csv(us_slim_path, index=False)

    print(f"Saved NY/TX slim dataset to: {ny_tx_slim_path}")
    print(f"Saved US aggregate slim dataset to: {us_slim_path}")


if __name__ == "__main__":
    main()


