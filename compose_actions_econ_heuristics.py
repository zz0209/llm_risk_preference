import os
import re
import json
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
    if len(parts) >= 5 and parts[1].startswith("W") and parts[-2] == "persona":
        state = parts[0]
        try:
            week = int(parts[1][1:])
        except Exception:
            return None
        su_id = parts[-1]
        return state, su_id, week
    if len(parts) >= 4 and parts[1].startswith("W") and parts[-2] == "persona":
        state = parts[0]
        try:
            week = int(parts[1][1:])
        except Exception:
            return None
        su_id = parts[-1]
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


def collect_actions(actions_dir: str) -> Dict[str, str]:
    data: Dict[str, str] = {}
    if not os.path.isdir(actions_dir):
        return data
    for fn in os.listdir(actions_dir):
        if not fn.endswith("_action.txt"):
            continue
        var = fn[:-len("_action.txt")]
        data[var] = read_text(os.path.join(actions_dir, fn))
    return data


def load_strata_knowledge(json_path: str) -> Dict:
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_code(text: str, pattern: str) -> Optional[int]:
    m = re.search(pattern, text, flags=re.IGNORECASE)
    if m:
        try:
            return int(m.group(1))
        except Exception:
            return None
    return None


def normalize(s: str) -> str:
    return re.sub(r"[\$,]", "", s.lower()).strip()


def find_hhincome_code(line: str, factor: Dict) -> Optional[int]:
    # Try to match option label substrings ignoring $, commas and case
    target = normalize(line)
    for opt in factor.get("options", []):
        label = normalize(opt.get("label", ""))
        if label and label in target:
            return int(opt.get("code"))
    # fallback: parse numeric bounds
    m = re.findall(r"(\d{1,3}(?:,\d{3})*)", line)
    if len(m) >= 1:
        try:
            vals = [int(x.replace(",", "")) for x in m]
            lo = min(vals)
            hi = max(vals)
            # crude bucketing by upper bound
            bounds = [10000, 20000, 30000, 40000, 50000, 75000, 100000, 150000]
            labels_to_code = {opt["label"]: opt["code"] for opt in factor.get("options", [])}
            # Map using known intervals
            if hi <= 10000:
                return 1
            if hi <= 20000:
                return 2
            if hi <= 30000:
                return 3
            if hi <= 40000:
                return 4
            if hi <= 50000:
                return 5
            if hi <= 75000:
                return 6
            if hi <= 100000:
                return 7
            if hi <= 150000:
                return 8
            return 9
        except Exception:
            return None
    return None


def build_strata_block(persona_text: str, rationales: Dict) -> str:
    # Extract codes from persona text
    age7_code = extract_code(persona_text, r"Age \(7-category\): \((\d)\)")
    pdense_code = extract_code(persona_text, r"Population density: \((\d)\)")
    edu_code = extract_code(persona_text, r"Highest education completed: \((\d)\)")
    # HHINCOME line
    hh_line_match = re.search(r"Household income:\s*([^\n\.]+)\.?", persona_text)
    hh_code = None
    if hh_line_match:
        hh_line = hh_line_match.group(0)
        hh_code = find_hhincome_code(hh_line, rationales["factors"]["HHINCOME"])

    out_lines: List[str] = []
    def add_factor(key: str, code: Optional[int]):
        fac = rationales["factors"].get(key, {})
        sal = fac.get("salience")
        opts = {int(o.get("code")): o for o in fac.get("options", []) if isinstance(o.get("code"), int)}
        opt = opts.get(code) if code is not None else None
        if opt:
            label = opt.get("label", "")
            rationale = opt.get("rationale", "")
            out_lines.append(f"- {key}={label} (salience {sal}): {rationale}")

    add_factor("AGE7", age7_code)
    add_factor("P_DENSE", pdense_code)
    add_factor("HHINCOME", hh_code)
    add_factor("EDUCATION", edu_code)
    return "\n".join(out_lines)


def build_perception_question() -> str:
    return (
        "You need to answer the following question as a single numeric probability.\n"
        "Question: Environment information above is macro statistics up to the past week. If you go to work, what is the probability that you will be infected with COVID-19?\n\n"
        "Answer format: Output one decimal number in [0,1] with EXACTLY three decimal places. "
        "Forbidden: rounding to or outputting values at multiples of 0.05 (e.g., 0.050, 0.100, 0.150, 0.200, ...). "
        "Return only the number; no text or symbols."
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--week", type=int, required=True, choices=[1,2,3])
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    persona_dir = os.path.join(base_dir, "prompts", "personas", f"W{args.week}")
    env_dir = os.path.join(base_dir, "prompts", "env")
    actions_dir = os.path.join(base_dir, "prompts", "actions")
    dag_template_path = os.path.join(base_dir, "prompts", "DAG", "template_env.txt")
    strata_json_path = os.path.join(base_dir, "prompts", "DAG", "stratification_rationales_AGE7_P_DENSE_HHINCOME_EDUCATION.txt")
    out_dir = os.path.join(base_dir, "prompts", "actions_econ_heuristics", f"week{args.week}")
    os.makedirs(out_dir, exist_ok=True)

    # Clean previous outputs
    for fn in list(os.listdir(out_dir)):
        if fn.endswith(".txt"):
            try:
                os.remove(os.path.join(out_dir, fn))
            except Exception:
                pass

    env_template = read_text(dag_template_path) if os.path.exists(dag_template_path) else ""
    strata = load_strata_knowledge(strata_json_path)
    env_map: Dict[str, str] = {}
    for st in ["NY", "TX"]:
        p = os.path.join(env_dir, f"{st}_W{args.week}_env.txt")
        env_map[st] = read_text(p) if os.path.exists(p) else ""

    personas = collect_personas(persona_dir)
    actions = collect_actions(actions_dir)
    # Restrict to ECON1 & ECON2; perception built-in
    keep_vars = {"ECON1", "ECON2"}

    total_written = 0
    for p in personas:
        state = p["state"]
        su_id = p["su_id"]
        persona_text = p["content"].strip()
        env_text = env_map.get(state, "").strip()
        if not env_text:
            continue

        strata_block = build_strata_block(persona_text, strata)

        def make_body(question_text: str, instructions_suffix: str) -> str:
            heur = (
                "Causal background (state-level):\n" + env_template.strip() + "\n\n" +
                "Stratified domain knowledge derived from persona:\n" + strata_block.strip()
            )
            return (
                "Your persona is described as follows:\n" + persona_text + "\n\n" +
                "Here is a summary of last week's pandemic snapshot:\n" + env_text + "\n\n" +
                heur + "\n\n" +
                "You need to do this survey question:\n" + question_text + instructions_suffix
            )

        # ECON1
        if "ECON1" in actions:
            body = make_body(actions["ECON1"].strip(), "\n\nAnswer option(s) as specified, with no explanation or anything else.")
            out_path = os.path.join(out_dir, f"{state}_W{args.week}_{su_id}_ECON1.txt")
            write_text(out_path, body)
            total_written += 1

        # ECON2 (numeric only)
        if "ECON2" in actions:
            instr = (
                "\n\nAnswer format for ECON2: Output a single integer number of total hours worked last week (0-168). "
                "Do NOT output 777, 998, or 999. Do NOT include any words, units, or punctuation. "
                "If uncertain, infer and provide your best estimate as an integer. Return only the number."
            )
            body = make_body(actions["ECON2"].strip(), instr)
            out_path = os.path.join(out_dir, f"{state}_W{args.week}_{su_id}_ECON2.txt")
            write_text(out_path, body)
            total_written += 1

        # PERCEPTION (risk perception question)
        per_q = build_perception_question()
        body = make_body(per_q, "")
        out_path = os.path.join(out_dir, f"{state}_W{args.week}_{su_id}_PERCEPTION.txt")
        write_text(out_path, body)
        total_written += 1

    print(f"Heuristics prompts written: {total_written} -> {out_dir}")


if __name__ == "__main__":
    main()


