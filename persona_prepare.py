import os
import re
import argparse
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


def normalize_mail50(val: str) -> str:
    if not isinstance(val, str):
        return ""
    # formats like "(3) 4/22/2020" or "4/22/2020"
    m = re.search(r"(\d{1,2})/(\d{1,2})/(\d{4})", val)
    if m:
        mm, dd, yyyy = m.group(1), m.group(2), m.group(3)
        return f"{int(mm):02d}-{int(dd):02d}-{yyyy}"
    m2 = re.search(r"(\d{1,2})-(\d{1,2})-(\d{4})", val)
    if m2:
        mm, dd, yyyy = m2.group(1), m2.group(2), m2.group(3)
        return f"{int(mm):02d}-{int(dd):02d}-{yyyy}"
    return ""


def process_week(base_dir: str, week: int) -> None:
    template_path = os.path.join(base_dir, "prompts", "personas", "template.txt")
    template_text = read_template(template_path)
    placeholders = extract_placeholders(template_text)

    out_dir = os.path.join(base_dir, "prompts", "personas", f"W{week}")
    os.makedirs(out_dir, exist_ok=True)

    for state, code in (("NY", "NY"), ("TX", "TX")):
        csv_path = os.path.join(base_dir, "survey", "survey_results", f"week{week}", f"Covid_W{week}_{state}.csv")
        df = pd.read_csv(csv_path, dtype=str)
        count = 0
        for _, r in df.iterrows():
            row = {k: (None if pd.isna(v) else v) for k, v in r.items()}
            su_id = row.get("SU_ID") or row.get("Su_ID") or row.get("Su_id")
            if not su_id:
                continue
            # Do not require MAIL50; filename without date
            content = render_template_for_row(template_text, placeholders, row)
            filename = f"{code}_W{week}_persona_{su_id}.txt"
            with open(os.path.join(out_dir, filename), "w", encoding="utf-8") as f:
                f.write(content)
            count += 1
        print(f"{state} W{week}: wrote {count} persona prompts -> {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--week", type=int, required=True, choices=[1, 2, 3])
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    process_week(base_dir, args.week)


if __name__ == "__main__":
    main()


