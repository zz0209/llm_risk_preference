import os
import re
import json
import time
import argparse
from typing import Iterable

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


def iter_prompt_files(root: str) -> Iterable[str]:
    if not os.path.isdir(root):
        return []
    for fn in os.listdir(root):
        if fn.endswith(".txt"):
            yield os.path.join(root, fn)


def detect_kind(filename: str) -> str:
    base = os.path.basename(filename)
    if base.endswith("_ECON1.txt"):
        return "ECON1"
    if base.endswith("_ECON2.txt"):
        return "ECON2"
    if base.endswith("_PERCEPTION.txt"):
        return "PERCEPTION"
    return "UNKNOWN"


def call_openai(api_key: str, model: str, prompt: str, timeout: float = 60.0) -> str:
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {"model": model, "messages": [{"role": "user", "content": prompt}], "temperature": 0}
    resp = requests.post(API_URL, headers=headers, data=json.dumps(payload), timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"].strip()


def is_valid_econ2(s: str) -> bool:
    s2 = s.strip()
    if not re.match(r"^\d+$", s2):
        return False
    v = int(s2)
    return 0 <= v <= 168


def is_valid_perception(s: str) -> bool:
    s2 = s.strip()
    if not re.match(r"^\d+\.\d{3}$", s2):
        return False
    v = float(s2)
    if not (0.0 <= v <= 1.0):
        return False
    # forbid multiples of 0.05
    if abs((v * 100) % 5) < 1e-9:
        return False
    return True


def main() -> None:
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set.")
    parser = argparse.ArgumentParser()
    parser.add_argument("--weeks", type=str, default="1,2,3")
    parser.add_argument("--model", type=str, default="gpt-4o")
    args = parser.parse_args()

    weeks = [int(w.strip()) for w in args.weeks.split(",") if w.strip()]

    base = os.path.dirname(os.path.abspath(__file__))
    in_root = os.path.join(base, "prompts", "actions_econ_heuristics")
    out_root = os.path.join(base, "outputs", "ECON_perception_heuritics")
    api_key = os.environ.get("OPENAI_API_KEY", "")

    for wk in weeks:
        src = os.path.join(in_root, f"week{wk}")
        dst = os.path.join(out_root, f"week{wk}")
        os.makedirs(dst, exist_ok=True)
        files = list(iter_prompt_files(src))
        files.sort()
        for path in tqdm(files, desc=f"Heuristics LLM W{wk}", unit="resp"):
            base = os.path.basename(path)
            kind = detect_kind(base)
            out_path = os.path.join(dst, base)
            if os.path.exists(out_path):
                continue
            prompt = read_text(path)
            ans = None
            for attempt in range(3):
                try:
                    ans = call_openai(api_key, args.model, prompt)
                except Exception:
                    time.sleep(1 + attempt)
                    continue
                if kind == "ECON2":
                    if is_valid_econ2(ans):
                        break
                elif kind == "PERCEPTION":
                    if is_valid_perception(ans):
                        break
                else:
                    # ECON1 or others, accept as-is
                    break
                # second try with stricter suffix
                prompt2 = prompt + ("\n\nReturn only the number." if kind == "ECON2" else "\n\nReturn only a 0-1 decimal with exactly three digits (e.g., 0.137). No words.")
                try:
                    ans = call_openai(api_key, args.model, prompt2)
                except Exception:
                    time.sleep(1 + attempt)
                    continue
                if kind == "ECON2" and is_valid_econ2(ans):
                    break
                if kind == "PERCEPTION" and is_valid_perception(ans):
                    break
            write_text(out_path, ans or "")

    print(f"Done. Wrote outputs to {out_root}")


if __name__ == "__main__":
    main()


