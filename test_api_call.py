import argparse
import os
import time
from typing import List

from openai import OpenAI


def read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def write_text(path: str, content: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


def list_first_n_prompts(final_dir: str, n: int) -> List[str]:
    files = [f for f in os.listdir(final_dir) if f.endswith(".txt")]
    files.sort()
    return [os.path.join(final_dir, f) for f in files[:n]]


def call_model(prompt: str, model: str = "gpt-4o", temperature: float = 0.0) -> str:
    client = OpenAI()
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
    )
    return resp.choices[0].message.content


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--model", type=str, default="gpt-4o")
    args = parser.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY 环境变量未设置。请先导出后重试。")

    base_dir = os.path.dirname(os.path.abspath(__file__))
    final_dir = os.path.join(base_dir, "prompts", "final", "W1")
    out_dir = os.path.join(base_dir, "outputs", "test")
    os.makedirs(out_dir, exist_ok=True)

    prompt_paths = list_first_n_prompts(final_dir, args.limit)
    print(f"Found {len(prompt_paths)} prompts. Starting API calls...")

    successes, failures = 0, 0
    for idx, p in enumerate(prompt_paths, 1):
        prompt_text = read_text(p)
        out_path = os.path.join(out_dir, os.path.basename(p))

        # Skip if already exists (idempotent)
        if os.path.exists(out_path):
            print(f"[{idx}/{len(prompt_paths)}] exists, skip: {out_path}")
            successes += 1
            continue

        # retry with simple backoff
        backoff = 2.0
        for attempt in range(5):
            try:
                output_text = call_model(prompt_text, model=args.model)
                write_text(out_path, output_text)
                print(f"[{idx}/{len(prompt_paths)}] wrote: {out_path}")
                successes += 1
                break
            except Exception as e:
                failures += 1 if attempt == 4 else 0
                print(f"[{idx}/{len(prompt_paths)}] error (attempt {attempt+1}): {e}")
                time.sleep(backoff)
                backoff *= 1.8

    print(f"Done. success={successes}, failures={failures}, outputs in {out_dir}")


if __name__ == "__main__":
    main()


