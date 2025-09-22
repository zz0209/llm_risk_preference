import os
import argparse
import pandas as pd
from datetime import datetime, timedelta


def load_state_data(csv_path: str, states):
    df = pd.read_csv(csv_path, dtype={"Province_State": str, "Report_Date": str})
    df = df[df["Province_State"].isin(states)].copy()
    df["Report_Date"] = pd.to_datetime(df["Report_Date"], errors="coerce")
    return df


def load_us_data(csv_path: str):
    df = pd.read_csv(csv_path, dtype={"Province_State": str, "Report_Date": str})
    df = df[df["Province_State"] == "US"].copy()
    df["Report_Date"] = pd.to_datetime(df["Report_Date"], errors="coerce")
    return df


def format_lines_for(df_state: pd.DataFrame, df_us: pd.DataFrame, state_name: str) -> str:
    lines = []
    lines.append(f"State: {state_name}")
    for _, row in df_state.iterrows():
        date_str = row["Report_Date"].strftime("%Y-%m-%d")
        lines.append(
            f"{date_str}: Confirmed={row.get('Confirmed')}, Deaths={row.get('Deaths')}, "
            f"Active={row.get('Active')}, CFR={row.get('Case_Fatality_Ratio')}, "
            f"Total_Test_Results={row.get('Total_Test_Results')}, People_Tested={row.get('People_Tested')}, "
            f"People_Hospitalized={row.get('People_Hospitalized')}"
        )
    lines.append("")
    lines.append("US Aggregate")
    for _, row in df_us.iterrows():
        date_str = row["Report_Date"].strftime("%Y-%m-%d")
        lines.append(
            f"{date_str}: Confirmed={row.get('Confirmed')}, Deaths={row.get('Deaths')}, "
            f"Active={row.get('Active')}, CFR={row.get('Case_Fatality_Ratio')}, "
            f"Total_Test_Results={row.get('Total_Test_Results')}, People_Tested={row.get('People_Tested')}, "
            f"People_Hospitalized={row.get('People_Hospitalized')}"
        )
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--week", type=int, required=True, choices=[1, 2, 3])
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    env_dir = os.path.join(base_dir, "prompts", "env")
    os.makedirs(env_dir, exist_ok=True)

    state_csv = os.path.join(base_dir, "data", "ny_tx_2020-04-10_to_2020-06-20_slim.csv")
    us_csv = os.path.join(base_dir, "data", "us_aggregate_2020-04-10_to_2020-06-20_slim.csv")

    # Define windows
    if args.week == 1:
        end_date = pd.Timestamp("2020-04-20")
        start_date = pd.Timestamp("2020-04-12")  # as in dataset start; we include all up to end
        header = "up to 2020-04-20 (inclusive)."
        suffix = "W1"
    elif args.week == 2:
        end_date = pd.Timestamp("2020-05-04")
        start_date = end_date - pd.Timedelta(days=8)  # prior 9 days inclusive
        header = "covering 9 days prior to 2020-05-04 (inclusive)."
        suffix = "W2"
    else:
        end_date = pd.Timestamp("2020-05-30")
        start_date = end_date - pd.Timedelta(days=6)  # prior 7 days inclusive
        header = "covering 7 days prior to 2020-05-30 (inclusive)."
        suffix = "W3"

    states_df = load_state_data(state_csv, ["New York", "Texas"])
    us_df_all = load_us_data(us_csv)

    # Filter windows
    states_df = states_df[(states_df["Report_Date"] >= start_date) & (states_df["Report_Date"] <= end_date)]
    us_df = us_df_all[(us_df_all["Report_Date"] >= start_date) & (us_df_all["Report_Date"] <= end_date)]

    ny_df = states_df[states_df["Province_State"] == "New York"].sort_values("Report_Date")
    tx_df = states_df[states_df["Province_State"] == "Texas"].sort_values("Report_Date")
    us_df = us_df.sort_values("Report_Date")

    ny_text = f"Context for New York {header}\n\n" + format_lines_for(ny_df, us_df, "New York")
    tx_text = f"Context for Texas {header}\n\n" + format_lines_for(tx_df, us_df, "Texas")

    with open(os.path.join(env_dir, f"NY_{suffix}_env.txt"), "w", encoding="utf-8") as f:
        f.write(ny_text)
    with open(os.path.join(env_dir, f"TX_{suffix}_env.txt"), "w", encoding="utf-8") as f:
        f.write(tx_text)

    print("Wrote prompts:")
    print(os.path.join(env_dir, f"NY_{suffix}_env.txt"))
    print(os.path.join(env_dir, f"TX_{suffix}_env.txt"))


if __name__ == "__main__":
    main()
