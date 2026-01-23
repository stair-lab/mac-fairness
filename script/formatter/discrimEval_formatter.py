# scripts/formatter/discrimEval_formatter.py
import os
import json
import argparse


def normalize_choices_to_abc(choices):
    """Normalize all choices to A/B/C format."""
    return [{"id": chr(65 + i), "text": str(choice)} for i, choice in enumerate(choices)]


def format_discrim_to_unified(input_path, output_path, subset):
    """
    Convert Discrim-Eval JSONL into unified JSONL format with:
    - question: filled_template
    - choices: Yes / No
    - correct_answer_id: always "A" 
    - source_id: running counter per line
    - question_id: discrimEval_<subset>_<source_id>
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
            correct_answer_id = "A"  # always A

            formatted = {
                "question_id": f"discrimEval_{subset}_{source_id}",
                "source_dataset": "DiscrimEval",
                "source_id": str(source_id),
                "question_type": "multiple_choice",
                "context": item.get("context", ""),
                "question": item.get("filled_template", ""),
                "choices": unified_choices,
                "correct_answer_id": correct_answer_id,
                "source_metadata": {
                    "additional_metadata": {**item},
                },
                "schema_version": "2025-12-10",
            }

            f_out.write(json.dumps(formatted) + "\n")
            running_id += 1


def process_all_discrim_files(input_dir, output_dir):
    """
    Traverse every .jsonl file in input_dir and format each one.
    Output filename matches input filename.
    """
    os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(input_dir):
        if not filename.endswith(".jsonl"):
            continue

        input_path = os.path.join(input_dir, filename)

        # subset = filename without .jsonl extension
        subset = os.path.splitext(filename)[0]

        output_filename = f"discrimEval_{subset}.jsonl"
        output_path = os.path.join(output_dir, output_filename)

        format_discrim_to_unified(input_path, output_path, subset)
        print(f"[DONE] {filename} -> {output_filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert all Discrim-Eval JSONL files in a directory into unified JSONL format"
    )

    parser.add_argument(
        "--input-dir",
        required=True,
        help="Directory containing DiscrimEval JSONL files (e.g., discrimEval_data/)",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where unified JSONL files should be saved",
    )

    args = parser.parse_args()

    process_all_discrim_files(args.input_dir, args.output_dir)
    print(f"\nAll Discrim-Eval files formatted and saved to: {args.output_dir}")

