import os
import pandas as pd


def filter_new_york_rows(df: pd.DataFrame, geo_col: str = "P_GEO") -> pd.DataFrame:
    if geo_col not in df.columns:
        return df.iloc[0:0]

    # Normalize to string for robust matching
    col = df[geo_col].astype(str)

    # Match either exact coded label or containing New York or code 8
    mask = (
        (col == "(8) New York")
        | col.str.contains("New York", case=False, na=False)
        | (col.str.strip() == "8")
    )
    return df[mask]


def process_week(base_dir: str, week: int) -> str:
    week_dir = os.path.join(base_dir, "survey", "survey_results", f"week{week}")
    input_csv = os.path.join(week_dir, f"Covid_W{week}_Full.csv")
    output_csv = os.path.join(week_dir, f"Covid_W{week}_NY.csv")

    if not os.path.exists(input_csv):
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")

    df = pd.read_csv(input_csv, dtype=str)
    ny_df = filter_new_york_rows(df, geo_col="P_GEO")
    ny_df.to_csv(output_csv, index=False)
    return output_csv


def main() -> None:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    outputs = []
    for week in (1, 2, 3):
        out = process_week(base_dir, week)
        outputs.append(out)
        print(f"W{week}: wrote -> {out}")


if __name__ == "__main__":
    main()


