import os
import re
import time
import argparse
from typing import List, Tuple

from tqdm import tqdm
import pandas as pd
from openai import OpenAI


def read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def write_text(path: str, content: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


def list_econ_prompts(dir_path: str, su_id: str, week_suffix: str) -> List[Tuple[str, str]]:
    # returns list of (variable, path)
    out = []
    for fn in os.listdir(dir_path):
        if not fn.endswith(".txt"):
            continue
        # {STATE}_{Wn}_{SU_ID}_{Variable}.txt
        m = re.match(rf"^[A-Z]{{2}}_{week_suffix}_([0-9]+)_([A-Z0-9_]+)\.txt$", fn)
        if not m:
            continue
        su = m.group(1)
        var = m.group(2)
        if su == su_id and var in {"ECON1", "ECON2", "ECON4"}:
            out.append((var, os.path.join(dir_path, fn)))
    # sort by variable
    out.sort(key=lambda x: x[0])
    return out


def call_model(prompt: str, model: str = "gpt-4o", temperature: float = 0.0) -> str:
    client = OpenAI()
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
    )
    return resp.choices[0].message.content.strip()


def main() -> None:
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set.")

    parser = argparse.ArgumentParser()
    parser.add_argument("--week", type=str, default="week1", choices=["week1", "week2", "week3"])
    parser.add_argument("--model", type=str, default="gpt-4o")
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    econ_prompts_dir = os.path.join(base_dir, "prompts", "actions_econ", args.week)
    out_dir = os.path.join(base_dir, "outputs", "ECON", args.week)
    os.makedirs(out_dir, exist_ok=True)

    # We will run for all SU_IDs present in the directory by scanning ECON1 files
    econ1_files = [f for f in os.listdir(econ_prompts_dir) if f.endswith("_ECON1.txt")]
    week_suffix = f"W{args.week.split('week')[-1]}" if args.week.startswith('week') else f"W{args.week}"
    pattern = re.compile(rf"^[A-Z]{{2}}_{week_suffix}_([0-9]+)_ECON1\.txt$")
    su_ids = sorted({pattern.match(f).group(1) for f in econ1_files if pattern.match(f)})

    total = 0
    for su_id in tqdm(su_ids, desc=f"{args.week} ECON LLM", unit="resp"):
        # call ECON1
        files = dict(list_econ_prompts(econ_prompts_dir, su_id, week_suffix))
        econ1_path = files.get("ECON1")
        if not econ1_path:
            continue
        econ1_out = os.path.join(out_dir, os.path.basename(econ1_path))
        if not os.path.exists(econ1_out):
            ans1 = call_model(read_text(econ1_path), model=args.model)
            write_text(econ1_out, ans1)
            total += 1
        else:
            ans1 = read_text(econ1_out)

        # decide next based on ECON1 answer code
        # Expect formats like "(1) ..." or containing "(1)"
        code = None
        m = re.search(r"\((\d+)\)", ans1)
        if m:
            code = int(m.group(1))

        if code == 1:
            # call ECON2 only
            econ2_path = files.get("ECON2")
            if econ2_path:
                econ2_out = os.path.join(out_dir, os.path.basename(econ2_path))
                if not os.path.exists(econ2_out):
                    prompt2 = read_text(econ2_path)
                    # first attempt
                    ans2 = call_model(prompt2, model=args.model)
                    # validate numeric-only 0-168, not 777/998/999
                    def valid_hours(s: str) -> bool:
                        s2 = s.strip()
                        if not s2.isdigit():
                            return False
                        v = int(s2)
                        return 0 <= v <= 168 and v not in {777, 998, 999}
                    if not valid_hours(ans2):
                        # retry with a brief, stricter reminder
                        ans2 = call_model(prompt2 + "\n\nReturn only an integer 0-168. Do not return 777/998/999 or any text.", model=args.model)
                    write_text(econ2_out, ans2)
                    total += 1
        elif code == 3:
            # call ECON4 only
            econ4_path = files.get("ECON4")
            if econ4_path:
                econ4_out = os.path.join(out_dir, os.path.basename(econ4_path))
                if not os.path.exists(econ4_out):
                    ans4 = call_model(read_text(econ4_path), model=args.model)
                    write_text(econ4_out, ans4)
                    total += 1
        else:
            # other ECON1 codes: do nothing else
            pass

    print(f"Done. Wrote/kept {total} outputs in {out_dir}")


if __name__ == "__main__":
    main()


