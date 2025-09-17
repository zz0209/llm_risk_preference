import os
import re
from typing import Dict, List, Tuple

import pandas as pd


def read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def list_llm_answers_for_su(outputs_dir: str, state: str, su_id: str) -> Dict[str, str]:
    pattern = re.compile(rf"^{state}_W1_{su_id}_(.+)\.txt$")
    llm_map: Dict[str, str] = {}
    for fn in os.listdir(outputs_dir):
        m = pattern.match(fn)
        if not m:
            continue
        variable = m.group(1)
        content = read_text(os.path.join(outputs_dir, fn)).strip()
        # 使用首个非空行作为答案（更稳健）
        first_line = next((line.strip() for line in content.splitlines() if line.strip()), "")
        llm_map[variable] = first_line if first_line else content
    return llm_map


def extract_code(value: str) -> str:
    if not isinstance(value, str):
        return ""
    m = re.search(r"\((\w+)\)", value)
    return m.group(1) if m else ""


def normalize(value: str) -> str:
    return re.sub(r"\s+", " ", str(value).strip()).lower()


def compare_answers(human: str, llm: str) -> bool:
    # 策略：优先比较括号编码，其次比较标准化字符串包含/相等
    hc, lc = extract_code(human), extract_code(llm)
    if hc and lc:
        return hc == lc
    hn, ln = normalize(human), normalize(llm)
    if hn == ln:
        return True
    if hn and hn in ln:
        return True
    if ln and ln in hn:
        return True
    return False


def main() -> None:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    outputs_dir = os.path.join(base_dir, "outputs", "test")
    human_csv = os.path.join(base_dir, "survey", "survey_results", "week1", "Covid_W1_NY.csv")
    out_dir = os.path.join(base_dir, "outputs", "compare")
    os.makedirs(out_dir, exist_ok=True)

    state = "NY"
    su_id = "0020001173"

    # 读取 LLM 作答
    llm_map = list_llm_answers_for_su(outputs_dir, state, su_id)

    # 读取人类答案（定位 SU_ID 行）
    df = pd.read_csv(human_csv, dtype=str)
    row = df[df["SU_ID"] == su_id]
    if row.empty:
        raise RuntimeError(f"Human row not found for SU_ID={su_id}")
    row = row.iloc[0].to_dict()

    # 共同问题集合（双方均有答案且非空）
    records: List[Dict[str, str]] = []
    for var, llm_ans in llm_map.items():
        human_ans = row.get(var)
        if human_ans is None:
            continue
        if pd.isna(human_ans) or str(human_ans).strip() == "":
            continue
        if str(llm_ans).strip() == "":
            continue
        is_match = compare_answers(human_ans, llm_ans)
        records.append(
            {
                "Variable": var,
                "Human_Answer": str(human_ans).strip(),
                "LLM_Answer": str(llm_ans).strip(),
                "Match": "Y" if is_match else "N",
            }
        )

    # 输出 CSV
    out_csv = os.path.join(out_dir, f"{state}_0020001173_vs_llm.csv")
    out_df = pd.DataFrame(records, columns=["Variable", "Human_Answer", "LLM_Answer", "Match"])
    out_df.sort_values("Variable").to_csv(out_csv, index=False)

    # 统计
    total = len(records)
    matched = sum(1 for r in records if r["Match"] == "Y")
    print(f"Compared {total} common questions. Matched={matched}, Mismatch={total - matched}.")
    print(f"Output: {out_csv}")


if __name__ == "__main__":
    main()



