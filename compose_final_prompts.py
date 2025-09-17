import os
import re
from typing import Dict, List


def read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def write_text(path: str, content: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


def parse_persona_filename(filename: str):
    # Expect: {STATE}_W1_{MAIL50}_persona_{SU_ID}.txt
    name, _ = os.path.splitext(filename)
    parts = name.split("_")
    if len(parts) < 5:
        return None
    state = parts[0]
    su_id = parts[-1]
    return state, su_id


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
    base_dir = os.path.dirname(os.path.abspath(__file__))
    persona_dir = os.path.join(base_dir, "prompts", "personas", "W1")
    env_dir = os.path.join(base_dir, "prompts", "env")
    actions_dir = os.path.join(base_dir, "prompts", "actions")
    out_dir = os.path.join(base_dir, "prompts", "final", "W1")
    os.makedirs(out_dir, exist_ok=True)

    # Load env texts
    env_map: Dict[str, str] = {
        "NY": read_text(os.path.join(env_dir, "NY_W1_env.txt")) if os.path.exists(os.path.join(env_dir, "NY_W1_env.txt")) else "",
        "TX": read_text(os.path.join(env_dir, "TX_W1_env.txt")) if os.path.exists(os.path.join(env_dir, "TX_W1_env.txt")) else "",
    }

    # Collect personas and actions
    personas = collect_personas(persona_dir)
    actions = collect_actions(actions_dir)

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

            final_text = (
                "Your persona is described as follows:\n"
                f"{persona_text}\n\n"
                "Here is a summary of last week's pandemic snapshot:\n"
                f"{env_text}\n\n"
                "You need to do this survey question:\n"
                f"{action_text}\n\n"
                "Answer option(s) as specified, with no explanation or anything else."
            )

            out_name = f"{state}_W1_{su_id}_{variable}.txt"
            out_path = os.path.join(out_dir, out_name)
            write_text(out_path, final_text)
            total_written += 1

    print(
        f"Final prompts written: {total_written} (personas={len(personas)} x actions={len(actions)}) to {out_dir}"
    )


if __name__ == "__main__":
    main()



