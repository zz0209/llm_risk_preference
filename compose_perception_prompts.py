import os
import argparse
from typing import Dict, List, Optional, Tuple


def read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def write_text(path: str, content: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


def parse_persona_filename(filename: str) -> Optional[Tuple[str, str, int]]:
    # Supports old: {STATE}_W{week}_{MAIL50}_persona_{SU_ID}.txt
    # and new:     {STATE}_W{week}_persona_{SU_ID}.txt
    name, _ = os.path.splitext(filename)
    parts = name.split("_")
    # Old format has 5 parts, new format 4 parts
    if len(parts) == 5 and parts[1].startswith("W") and parts[2] != "persona" and parts[3] == "persona":
        state = parts[0]
        try:
            week = int(parts[1][1:])
        except Exception:
            return None
        su_id = parts[4]
        return state, su_id, week
    if len(parts) == 4 and parts[1].startswith("W") and parts[2] == "persona":
        state = parts[0]
        try:
            week = int(parts[1][1:])
        except Exception:
            return None
        su_id = parts[3]
        return state, su_id, week
    return None


def collect_personas(persona_dir: str) -> List[Dict[str, str]]:
    items: List[Dict[str, str]] = []
    if not os.path.isdir(persona_dir):
        return items
    for fn in os.listdir(persona_dir):
        if not fn.endswith(".txt"):
            continue
        parsed = parse_persona_filename(fn)
        if not parsed:
            continue
        state, su_id, week = parsed
        content = read_text(os.path.join(persona_dir, fn))
        items.append({"state": state, "su_id": su_id, "week": str(week), "content": content})
    return items


def env_text_for(state: str, week: int, env_dir: str) -> str:
    path = os.path.join(env_dir, f"{state}_W{week}_env.txt")
    return read_text(path) if os.path.exists(path) else ""


def build_perception_prompt(persona_text: str, env_text: str) -> str:
    instructions = (
        "You need to answer the following question as a single numeric probability.\n"
        "Question: Environment information above is macro statistics up to the past week. If you go to work, what is the probability that you will be infected with COVID-19?\n\n"
        "Answer format: Output one decimal number in [0,1] with EXACTLY three decimal places. "
        "Forbidden: rounding to or outputting values at multiples of 0.05 (e.g., 0.050, 0.100, 0.150, 0.200, ...). "
        "Return only the number; no text or symbols."
    )
    txt = (
        "Your persona is described as follows:\n"
        f"{persona_text.strip()}\n\n"
        "Here is a summary of last week's pandemic snapshot:\n"
        f"{env_text.strip()}\n\n"
        f"{instructions}"
    )
    return txt


def process_all(base_dir: str, weeks: List[int]) -> None:
    env_dir = os.path.join(base_dir, "prompts", "env")
    out_dir = os.path.join(base_dir, "prompts", "perception")
    total = 0
    for week in weeks:
        persona_dir = os.path.join(base_dir, "prompts", "personas", f"W{week}")
        personas = collect_personas(persona_dir)
        if not personas:
            print(f"No personas found for W{week} in {persona_dir}")
            continue
        for p in personas:
            state = p["state"]
            su_id = p["su_id"]
            persona_text = p["content"]
            env_text = env_text_for(state, week, env_dir)
            if not env_text:
                # skip if no env
                continue
            final_text = build_perception_prompt(persona_text, env_text)
            out_name = f"{state}_W{week}_{su_id}.txt"
            out_path = os.path.join(out_dir, out_name)
            write_text(out_path, final_text)
            total += 1
    print(f"Perception prompts written: {total} -> {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weeks", type=str, default="1,2,3")
    args = parser.parse_args()
    weeks = []
    for w in args.weeks.split(","):
        w = w.strip()
        if not w:
            continue
        try:
            weeks.append(int(w))
        except Exception:
            pass
    if not weeks:
        weeks = [1, 2, 3]
    base_dir = os.path.dirname(os.path.abspath(__file__))
    process_all(base_dir, weeks)


if __name__ == "__main__":
    main()


