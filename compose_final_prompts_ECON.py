import os
import re
import argparse
from typing import Dict, List


def read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def write_text(path: str, content: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


def parse_persona_filename(filename: str):
    # Support both formats:
    # 1) {STATE}_W{week}_{MAIL50}_persona_{SU_ID}.txt
    # 2) {STATE}_W{week}_persona_{SU_ID}.txt
    name, _ = os.path.splitext(filename)
    parts = name.split("_")
    if len(parts) >= 5 and parts[-2] == "persona":
        state = parts[0]
        su_id = parts[-1]
        return state, su_id
    if len(parts) >= 4 and parts[-2] == "persona":
        state = parts[0]
        su_id = parts[-1]
        return state, su_id
    return None


def collect_personas(persona_dir: str) -> List[Dict[str, str]]:
    items = []
    for fn in os.listdir(persona_dir):
        if not fn.endswith(".txt"):
            continue
        parsed = parse_persona_filename(fn)
        if not parsed:
            continue
        state, su_id = parsed
        content = read_text(os.path.join(persona_dir, fn))
        items.append({"state": state, "su_id": su_id, "content": content})
    return items


def collect_actions(actions_dir: str) -> List[Dict[str, str]]:
    items = []
    for fn in os.listdir(actions_dir):
        if not fn.endswith("_action.txt"):
            continue
        variable = fn[:-len("_action.txt")]
        content = read_text(os.path.join(actions_dir, fn))
        items.append({"variable": variable, "content": content})
    return items


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--week", type=int, required=True, choices=[1,2,3])
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    persona_dir = os.path.join(base_dir, "prompts", "personas", f"W{args.week}")
    env_dir = os.path.join(base_dir, "prompts", "env")
    actions_dir = os.path.join(base_dir, "prompts", "actions")
    out_dir = os.path.join(base_dir, "prompts", "actions_econ", f"week{args.week}")
    os.makedirs(out_dir, exist_ok=True)

    # Clean previous outputs to avoid stale files (e.g., ECON3)
    for fn in list(os.listdir(out_dir)):
        if fn.endswith(".txt"):
            try:
                os.remove(os.path.join(out_dir, fn))
            except Exception:
                pass

    # Load env texts
    suffix = f"W{args.week}"
    env_map: Dict[str, str] = {
        "NY": read_text(os.path.join(env_dir, f"NY_{suffix}_env.txt")) if os.path.exists(os.path.join(env_dir, f"NY_{suffix}_env.txt")) else "",
        "TX": read_text(os.path.join(env_dir, f"TX_{suffix}_env.txt")) if os.path.exists(os.path.join(env_dir, f"TX_{suffix}_env.txt")) else "",
    }

    # Collect personas and actions
    personas = collect_personas(persona_dir)
    all_actions = collect_actions(actions_dir)
    # Only keep ECON1, ECON2, ECON4
    actions = [a for a in all_actions if a["variable"] in {"ECON1", "ECON2", "ECON4"}]

    total_written = 0
    for p in personas:
        state = p["state"]
        su_id = p["su_id"]
        persona_text = p["content"].strip()

        # Map state label used in persona filenames to env key
        env_key = state
        env_text = env_map.get(env_key, "").strip()
        if not env_text:
            # Skip if no env available for this state
            continue

        for a in actions:
            variable = a["variable"]
            action_text = a["content"].strip()

            if variable == "ECON2":
                # Enforce numeric-only hours answer
                instructions = (
                    "\n\nAnswer format for ECON2: Output a single integer number of total hours worked last week (0-168). "
                    "Do NOT output 777, 998, or 999. Do NOT include any words, units, or punctuation. "
                    "If uncertain, infer and provide your best estimate as an integer. Return only the number."
                )
            else:
                instructions = "\n\nAnswer option(s) as specified, with no explanation or anything else."

            final_text = (
                "Your persona is described as follows:\n"
                f"{persona_text}\n\n"
                "Here is a summary of last week's pandemic snapshot:\n"
                f"{env_text}\n\n"
                "You need to do this survey question:\n"
                f"{action_text}{instructions}"
            )

            out_name = f"{state}_W{args.week}_{su_id}_{variable}.txt"
            out_path = os.path.join(out_dir, out_name)
            write_text(out_path, final_text)
            total_written += 1

    print(
        f"ECON prompts written: {total_written} (personas={len(personas)} x actions={len(actions)}) to {out_dir}"
    )


if __name__ == "__main__":
    main()



