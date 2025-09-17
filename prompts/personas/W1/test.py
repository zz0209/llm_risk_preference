import os
import re
import sys

import pandas as pd


def normalize_mail50(val):
    if not isinstance(val, str):
        return ""
    m = re.search(r"(\d{1,2})/(\d{1,2})/(\d{4})", val)
    if m:
        mm, dd, yyyy = m.group(1), m.group(2), m.group(3)
        return f"{int(mm):02d}-{int(dd):02d}-{yyyy}"
    m2 = re.search(r"(\d{1,2})-(\d{1,2})-(\d{4})", val)
    if m2:
        mm, dd, yyyy = m2.group(1), m2.group(2), m2.group(3)
        return f"{int(mm):02d}-{int(dd):02d}-{yyyy}"
    return ""


def main():
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    week1_dir = os.path.join(repo_root, "survey", "survey_results", "week1")
    out_dir = os.path.join(repo_root, "prompts", "personas", "W1")

    ny_csv = os.path.join(week1_dir, "Covid_W1_NY.csv")
    tx_csv = os.path.join(week1_dir, "Covid_W1_TX.csv")

    df_ny = pd.read_csv(ny_csv, dtype=str)
    df_tx = pd.read_csv(tx_csv, dtype=str)

    def count_blank_mail50(df):
        cnt = 0
        for v in df.get("MAIL50", []):
            if normalize_mail50(v) == "":
                cnt += 1
        return cnt

    total_rows = len(df_ny) + len(df_tx)
    blank_mail50 = count_blank_mail50(df_ny) + count_blank_mail50(df_tx)
    expected_files = total_rows - blank_mail50

    actual_files = [f for f in os.listdir(out_dir) if f.endswith(".txt")]
    actual_count = len(actual_files)

    print(f"NY rows={len(df_ny)}, TX rows={len(df_tx)}")
    print(f"MAIL50 blanks total={blank_mail50}")
    print(f"Expected files={expected_files}, Actual files={actual_count}")

    if actual_count != expected_files:
        print("Mismatch: counts do not align", file=sys.stderr)
        sys.exit(1)
    print("Test passed")


if __name__ == "__main__":
    main()



