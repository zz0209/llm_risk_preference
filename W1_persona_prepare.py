import os
import re
from typing import Dict, List, Set

import pandas as pd


def read_template(template_path: str) -> str:
    with open(template_path, "r", encoding="utf-8") as f:
        return f.read()


def extract_placeholders(template_text: str) -> Set[str]:
    return set(re.findall(r"\{([A-Z0-9_]+)\}", template_text))


def render_template_for_row(template_text: str, placeholders: Set[str], row: Dict[str, str]) -> str:
    rendered = template_text
    for key in placeholders:
        value = row.get(key)
        if value is None or pd.isna(value):
            value_str = "NA"
        else:
            value_str = str(value)
        rendered = rendered.replace("{" + key + "}", value_str)
    return rendered


def normalize_mail50(mail50_value: str) -> str:
    if not isinstance(mail50_value, str):
        return ""
    # Expected like "(3) 4/22/2020"; extract the date part
    m = re.search(r"(\d{1,2})/(\d{1,2})/(\d{4})", mail50_value)
    if not m:
        # Also handle already formatted like 04-22-2020
        m2 = re.search(r"(\d{1,2})-(\d{1,2})-(\d{4})", mail50_value)
        if m2:
            mm, dd, yyyy = m2.group(1), m2.group(2), m2.group(3)
            return f"{int(mm):02d}-{int(dd):02d}-{yyyy}"
        return ""
    mm, dd, yyyy = m.group(1), m.group(2), m.group(3)
    return f"{int(mm):02d}-{int(dd):02d}-{yyyy}"


def process_csv(input_csv_path: str, template_text: str, placeholders: Set[str], state_code: str, output_dir: str) -> int:
    df = pd.read_csv(input_csv_path, dtype=str)
    count_written = 0
    for _, row_series in df.iterrows():
        row: Dict[str, str] = row_series.to_dict()

        mail50_raw = row.get("MAIL50")
        mail50_formatted = normalize_mail50(mail50_raw) if mail50_raw not in (None, "", "NA", "NaN", "nan") else ""
        if not mail50_formatted:
            # Skip rows without usable MAIL50
            continue

        su_id = row.get("SU_ID")
        if su_id is None or pd.isna(su_id) or str(su_id).strip() == "":
            # If SU_ID missing, skip to avoid unnamed files
            continue
        su_id = str(su_id)

        rendered = render_template_for_row(template_text, placeholders, row)

        file_name = f"{state_code}_W1_{mail50_formatted}_persona_{su_id}.txt"
        output_path = os.path.join(output_dir, file_name)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(rendered)
        count_written += 1
    return count_written


def main() -> None:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    week1_dir = os.path.join(base_dir, "survey", "survey_results", "week1")
    template_path = os.path.join(base_dir, "prompts", "personas", "template.txt")
    output_dir = os.path.join(base_dir, "prompts", "personas", "W1")
    os.makedirs(output_dir, exist_ok=True)

    template_text = read_template(template_path)
    placeholders = extract_placeholders(template_text)

    jobs: List[Dict[str, str]] = [
        {
            "csv": os.path.join(week1_dir, "Covid_W1_NY.csv"),
            "state": "NY",
        },
        {
            "csv": os.path.join(week1_dir, "Covid_W1_TX.csv"),
            "state": "TX",
        },
    ]

    total = 0
    for job in jobs:
        written = process_csv(
            input_csv_path=job["csv"],
            template_text=template_text,
            placeholders=placeholders,
            state_code=job["state"],
            output_dir=output_dir,
        )
        print(f"{job['state']}: wrote {written} persona prompts")
        total += written
    print(f"Total prompts written: {total}")


if __name__ == "__main__":
    main()



