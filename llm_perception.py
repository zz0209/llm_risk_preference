import os
import re
import time
import json
import argparse
from typing import Iterable, Tuple, Optional

import requests
from tqdm import tqdm


API_URL = "https://api.openai.com/v1/chat/completions"


def read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def write_text(path: str, content: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


def iter_perception_prompts(root: str) -> Iterable[str]:
    if not os.path.isdir(root):
        return []
    for fn in os.listdir(root):
        if fn.endswith(".txt"):
            yield os.path.join(root, fn)


def parse_week_and_ids(filename: str) -> Tuple[Optional[int], Optional[str], Optional[str]]:
    # {STATE}_W{week}_{SU_ID}.txt
    base = os.path.basename(filename)
    m = re.match(r"^([A-Z]{2})_W(\d)_([0-9]+)\.txt$", base)
    if not m:
        return None, None, None
    state, w, su_id = m.group(1), m.group(2), m.group(3)
    try:
        week = int(w)
    except Exception:
        return None, None, None
    return week, state, su_id


def call_openai_chat(api_key: str, model: str, prompt: str, timeout: float = 60.0) -> str:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.2,
    }
    resp = requests.post(API_URL, headers=headers, data=json.dumps(payload), timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"].strip()


def is_valid_prob(s: str) -> bool:
    if not isinstance(s, str):
        return False
    s2 = s.strip()
    if not re.match(r"^\d*\.?\d+$", s2):
        return False
    try:
        v = float(s2)
    except Exception:
        return False
    return 0.0 <= v <= 1.0


def extract_prob_anywhere(s: str) -> Optional[str]:
    if not isinstance(s, str):
        return None
    # find first number in [0,1]
    for m in re.finditer(r"(?<!\d)(?:0(?:\.\d+)?|1(?:\.0+)?)\b", s):
        cand = m.group(0)
        try:
            v = float(cand)
            if 0.0 <= v <= 1.0:
                return cand
        except Exception:
            continue
    return None


def main() -> None:
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set.")

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="gpt-4o")
    parser.add_argument("--weeks", type=str, default="1,2,3")
    parser.add_argument("--max_retries", type=int, default=2)
    args = parser.parse_args()

    allowed_weeks = set()
    for w in args.weeks.split(","):
        w = w.strip()
        if not w:
            continue
        try:
            allowed_weeks.add(int(w))
        except Exception:
            pass
    if not allowed_weeks:
        allowed_weeks = {1, 2, 3}

    base_dir = os.path.dirname(os.path.abspath(__file__))
    in_dir = os.path.join(base_dir, "prompts", "perception")
    out_root = os.path.join(base_dir, "outputs", "perception")

    files = list(iter_perception_prompts(in_dir))
    # filter by week
    files = [p for p in files if parse_week_and_ids(p)[0] in allowed_weeks]
    files.sort()

    api_key = os.environ.get("OPENAI_API_KEY", "")
    written = 0
    for p in tqdm(files, desc="Perception LLM", unit="resp"):
        week, state, su_id = parse_week_and_ids(p)
        if week is None:
            continue
        out_dir = os.path.join(out_root, f"week{week}")
        out_path = os.path.join(out_dir, os.path.basename(p))
        if os.path.exists(out_path):
            # cache
            continue
        prompt = read_text(p)
        # call with validation and retries
        ans = None
        for attempt in range(args.max_retries + 1):
            try:
                ans = call_openai_chat(api_key, args.model, prompt)
            except Exception:
                time.sleep(1.0 + attempt)
                continue
            if is_valid_prob(ans):
                break
            # Second chance: stricter instruction
            if attempt < args.max_retries:
                strict = prompt + "\n\nReturn only a numeric value between 0 and 1 (e.g., 0.35). No text, no symbols."
                try:
                    ans = call_openai_chat(api_key, args.model, strict)
                except Exception:
                    time.sleep(1.0 + attempt)
                    continue
                if is_valid_prob(ans):
                    break
        if not is_valid_prob(ans or ""):
            # Try to extract first valid numeric piece from the text
            extracted = extract_prob_anywhere(ans or "")
            if extracted is not None:
                ans = extracted
        write_text(out_path, ans or "")
        written += 1

    print(f"Done. Wrote/kept {written} outputs in {out_root}")


if __name__ == "__main__":
    main()


