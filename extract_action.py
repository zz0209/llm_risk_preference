import json
import os


def main() -> None:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    questionnaire_path = os.path.join(base_dir, "survey", "survey_results", "final_questionnaire.json")
    out_dir = os.path.join(base_dir, "prompts", "actions")
    os.makedirs(out_dir, exist_ok=True)

    with open(questionnaire_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    action_items = [item for item in data if item.get("Category") == "action"]

    for item in action_items:
        var = item.get("Variable")
        
        # Dump the exact JSON content in a readable txt form
        content = json.dumps(item, ensure_ascii=False, indent=2)
        filename = f"{var}_action.txt"
        out_path = os.path.join(out_dir, filename)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(content)
    print(f"Wrote {len(action_items)} action prompts to {out_dir}")


if __name__ == "__main__":
    main()


