# script/formatter/discrimEval_formatter.py
import json
import argparse


def normalize_choices_to_abc(choices):
    """Normalize all choices to A/B/C format.

    Binary choices (yes/no, true/false, etc.) always map to A/B.
    Multiple choices map to A/B/C/D/etc.
    """
    return [
        {"id": chr(65 + i), "text": str(choice)} for i, choice in enumerate(choices)
    ]


def format_discrim_to_unified(input_path, output_path, subset):
    """
    Convert Discrim-Eval JSONL into unified JSONL format with:

    - question: filled_template
    - choices: Yes / No
    - correct_answer_id: null
    - source_id: running counter per line
    - question_id: discrim_<subset>_<running_id>_<race>
    """

    # Fixed binary choices
    base_choices = ["Yes", "No"]
    unified_choices = normalize_choices_to_abc(base_choices)

    running_id = 0 

    with open(input_path, "r") as f_in, open(output_path, "w") as f_out:

        for line in f_in:
            if not line.strip():
                continue

            item = json.loads(line)

            source_id = running_id
            correct_answer_id = "A"

            formatted = {
                # Unified question_id: benchmark_subcategory_originalid
                "question_id": f"discrimEval_{subset}_{source_id}",
                "source_dataset": "DiscrimEval",
                "source_id": str(source_id),
                "question_type": "multiple_choice",
                "context": item.get("context", ""),
                "question": item.get("filled_template", ""),
                "choices": unified_choices,  # Always A/B/C format
                "correct_answer_id": correct_answer_id,  # Always A/B/C format
                # Preserve metadata
                "source_metadata": {
                    "example_id": item.get("example_id"),
                    "question_index": item.get("question_index"),
                    "question_polarity": item.get("question_polarity"),
                    "context_condition": item.get("context_condition"),
                    "category": item.get("category"),
                    "answer_info": item.get("answer_info"),
                    "label": item.get("label"),
                    # Store original choice fields for reference
                    "ans0": item.get("ans0"),
                    "ans1": item.get("ans1"),
                    "ans2": item.get("ans2"),
                    # Preserve any additional metadata
                    "additional_metadata": {
                        **item,
                    },
                },
                "schema_version": "2025-12-10",
            }

            f_out.write(json.dumps(formatted) + "\n")

            running_id += 1  


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert Discrim-Eval JSONL (explicit/implicit) to unified JSONL format"
    )
    parser.add_argument(
        "--input", required=True, help="Path to input discrim-eval JSONL file (e.g., explicit.jsonl)",
    )
    parser.add_argument(
        "--output", required=True, help="Path to output unified JSONL file",
    )
    parser.add_argument(
        "--subset", default="explicit", help="Subset name used in question_id (e.g., explicit, implicit)",
    )

    args = parser.parse_args()
    format_discrim_to_unified(args.input, args.output, args.subset)
    print(f"Formatted Discrim-Eval ({args.subset}) data saved to {args.output}")
