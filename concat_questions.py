import json
import os
from typing import Dict, List


def load_items(json_path: str) -> List[dict]:
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_variable_to_item(items: List[dict]) -> Dict[str, dict]:
    mapping: Dict[str, dict] = {}
    for item in items:
        variable = item.get("Variable")
        if isinstance(variable, str):
            mapping[variable] = item
    return mapping


def classify_category(variable: str, label: str) -> str:
    """Return 'persona' or 'action' based on Variable code/label heuristics."""
    if not isinstance(variable, str):
        return "persona"

    # Explicit action prefixes/categories (decision or taken measures)
    action_prefixes = (
        "PHYS2_",  # measures taken in response to coronavirus
        "PHYS10",  # willingness to participate
        "ECON6",   # received/applied/tried to apply assistance
        "ECON7",   # how would you pay (decision)
        "ECON8",   # plans changed due to restrictions
    )
    if any(variable.startswith(p) for p in action_prefixes):
        return "action"

    # Specific action variables
    if variable in {"ECON1"}:
        return "action"

    # Everything else is treated as background/persona
    return "persona"


def main() -> None:
    base_dir = os.path.dirname(os.path.abspath(__file__))

    week1_path = os.path.join(
        base_dir,
        "survey",
        "survey_results",
        "week1",
        "COVID_Impact_Codebook_v1_3_items_grouped.json",
    )
    week2_path = os.path.join(
        base_dir,
        "survey",
        "survey_results",
        "week2",
        "COVID_Impact_Codebook_v2_2_items_grouped.json",
    )
    week3_path = os.path.join(
        base_dir,
        "survey",
        "survey_results",
        "week3",
        "COVID_Impact_Codebook_items_grouped.json",
    )

    # Load JSON arrays
    items_week1 = load_items(week1_path)
    items_week2 = load_items(week2_path)
    items_week3 = load_items(week3_path)

    # Build mappings
    map_w1 = build_variable_to_item(items_week1)
    map_w2 = build_variable_to_item(items_week2)
    map_w3 = build_variable_to_item(items_week3)

    # Compute intersection by Variable ID
    common_variables = set(map_w1.keys()) & set(map_w2.keys()) & set(map_w3.keys())

    # Preserve Week 1 order for determinism
    ordered_common_items: List[dict] = []
    for item in items_week1:
        var = item.get("Variable")
        if var in common_variables:
            # Prefer Week 1 representation; add Category classification
            base_item = dict(map_w1[var])
            base_item["Category"] = classify_category(
                var, base_item.get("Variable Label", "")
            )
            ordered_common_items.append(base_item)

    # Output path under survey_results directory (as requested)
    output_path = os.path.join(base_dir, "survey", "survey_results", "final_questionnaire.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(ordered_common_items, f, ensure_ascii=False, indent=2)

    print(
        f"Wrote {len(ordered_common_items)} common items (out of W1={len(map_w1)}, W2={len(map_w2)}, W3={len(map_w3)}) to: {output_path}"
    )


if __name__ == "__main__":
    main()


