import os
import pandas as pd


def filter_texas_rows(df: pd.DataFrame, geo_col: str = "P_GEO") -> pd.DataFrame:
    if geo_col not in df.columns:
        return df.iloc[0:0]

    col = df[geo_col].astype(str)
    mask = (
        (col == "(10) Texas")
        | col.str.contains("Texas", case=False, na=False)
        | (col.str.strip() == "10")
    )
    return df[mask]


def process_week(base_dir: str, week: int) -> str:
    week_dir = os.path.join(base_dir, "survey", "survey_results", f"week{week}")
    input_csv = os.path.join(week_dir, f"Covid_W{week}_Full.csv")
    output_csv = os.path.join(week_dir, f"Covid_W{week}_TX.csv")

    if not os.path.exists(input_csv):
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")

    df = pd.read_csv(input_csv, dtype=str)
    tx_df = filter_texas_rows(df, geo_col="P_GEO")
    tx_df.to_csv(output_csv, index=False)
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


