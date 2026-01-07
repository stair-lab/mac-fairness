# script/formatter/diffAware_formatter.py
import json
import argparse
import pickle


def normalize_choices_to_abc(choices):
    """Normalize all choices to A/B/C format.

    Binary choices (yes/no, true/false, etc.) always map to A/B.
    Multiple choices map to A/B/C/D/etc.
    """
    return [
        {"id": chr(65 + i), "text": str(choice)} for i, choice in enumerate(choices)
    ]

def split_question_and_choices(full_text):
    """Split question into question-only text and choice texts"""
    if "\n" not in full_text:
        return full_text, ["", "", ""]   # No choices found

    question_part, choices_part = full_text.split("\n", 1)

    # Split for (a), (b), (c) 
    try:
        a = choices_part.split("(a)")[1].split("(b)")[0].strip()
        b = choices_part.split("(b)")[1].split("(c)")[0].strip()
        c = choices_part.split("(c)")[1].strip()
        return question_part.strip(), [a, b, c]
    except:
        # If parsing fails, leave question whole with blank choices
        return question_part.strip(), ["", "", ""]


def load_df0_pkl_as_json_lines(input_pkl_path):
    """
    Convert .pkl dataset to 'JSON lines'
    
    .pkl structure:
        [different, equal]
    where each element is:
        [question_str, label_int, question_id]

    """
    with open(input_pkl_path, "rb") as f:
        loaded = pickle.load(f)

    json_lines = []

    # Should expect [different, equal]; if not, just treat as a flat list
    if isinstance(loaded, (list, tuple)) and len(loaded) == 2:
        splits = [("different", loaded[0]), ("equal", loaded[1])]
    else:
        splits = [("all", loaded)]

    for split_name, split_data in splits:
        for record in split_data:
            # record: [question_str, label_int, question_id]
            if not isinstance(record, (list, tuple)) or len(record) < 3:
                continue

            question_str, label_int, example_id = record[0], record[1], record[2]
            # Get just the question and its choices separately
            question, choices = split_question_and_choices(question_str)

            item = {
                "question": question,
                "ans0": choices[0],
                "ans1": choices[1],
                "ans2": choices[2],
                "label": label_int,
                "example_id": example_id,
                "context": None,
                "question_index": None,
                "question_polarity": None,
                "context_condition": None,
                "category": None,
                "answer_info": None,
                "additional_metadata": {
                    "split": split_name
                },
            }

            json_lines.append(json.dumps(item))

    return json_lines


def format_diffAware_to_unified(input_path, output_path, dataset_name="D1_1K"):
    """
      
    """
    # Convert .pkl to JSON
    json_lines = load_df0_pkl_as_json_lines(input_path)

    with open(output_path, "w") as f_out:
        for idx, line in enumerate(json_lines):
            if not line.strip():
                continue

            item = json.loads(line)

            # Extract choices (ans0, ans1, ans2)
            choices = []
            for i in range(3):
                choice_key = f"ans{i}"
                if choice_key in item:
                    choices.append(item[choice_key])

            # Map to A/B/C format
            unified_choices = [
                {"id": chr(65 + i), "text": choice} for i, choice in enumerate(choices)
            ]

            # Map label (0,1,2) to answer ID (A,B,C)
            label = item.get("label")
            correct_answer_id = chr(65 + label) if label is not None else None

            # Build the minimal unified format
            formatted = {
                # Unified question_id: benchmark_subcategory_originalid
                "question_id": f"{dataset_name}_{item['additional_metadata']['split']}_{idx}",
                "source_dataset": "DiffAware",
                "source_id": str(item.get("example_id", idx)),
                "question_type": "multiple_choice",
                "context": item.get("context", ""),
                "question": item["question"],
                "choices": unified_choices,  # Always A/B/C format
                "correct_answer_id": correct_answer_id,  # Always A/B/C format
                # Preserve "metadata"; fields not in .pkl will return Null
                "source_metadata": {
                    "example_id": item.get("example_id"),
                    "question_index": item.get("question_index"),
                    "question_polarity": item.get("question_polarity"),
                    "context_condition": item.get("context_condition"),
                    "category": item.get("category"),
                    "answer_info": item.get("answer_info"),
                    "label": item.get("label"),
                    "ans0": None,
                    "ans1": None,
                    "ans2": None,
                    # Preserve any additional metadata
                    "additional_metadata": item.get("additional_metadata"),
                },
                "schema_version": "2025-12-10",
            }

            f_out.write(json.dumps(formatted) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert D1_1K .pkl benchmark to unified JSONL format"
    )
    parser.add_argument(
        "--input", required=True, help="Path to input diffAware subcategory .pkl file"
    )
    parser.add_argument(
        "--output", required=True, help="Path to output unified JSONL file"
    )
    parser.add_argument(
        "--dataset_name", default="D1_1K", help="Dataset name (e.g. D1_1K)",
    )

    args = parser.parse_args()
    format_diffAware_to_unified(args.input, args.output, args.dataset_name)
    print(f"Formatted {args.dataset_name} data saved to {args.output}")
