import json
import os
import re


def load_questionnaire(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_template(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def extract_placeholders(template_text: str):
    # placeholders are of form {VARIABLE}
    return set(re.findall(r"\{([A-Z0-9_]+)\}", template_text))


def main():
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    questionnaire_path = os.path.join(
        repo_root, "survey", "survey_results", "final_questionnaire.json"
    )
    template_path = os.path.join(
        repo_root, "prompts", "personas", "template.txt"
    )

    questionnaire = load_questionnaire(questionnaire_path)
    template_text = load_template(template_path)

    placeholders = extract_placeholders(template_text)
    persona_vars = set(
        item.get("Variable")
        for item in questionnaire
        if item.get("Category") == "persona" and isinstance(item.get("Variable"), str)
    )

    # 1) Ensure all placeholders exist in questionnaire persona variables
    missing_in_questionnaire = sorted([v for v in placeholders if v not in persona_vars])

    # 2) Ensure all persona variables are covered by template
    missing_in_template = sorted([v for v in persona_vars if v not in placeholders])

    print("Template placeholders count:", len(placeholders))
    print("Persona variables count:", len(persona_vars))
    print("Placeholders not found in questionnaire persona:")
    for v in missing_in_questionnaire:
        print("  -", v)

    print("Persona variables missing in template:")
    for v in missing_in_template:
        print("  -", v)

    # Non-zero exit to signal failures
    if missing_in_questionnaire or missing_in_template:
        raise SystemExit(1)
    else:
        print("All checks passed.")


if __name__ == "__main__":
    main()


