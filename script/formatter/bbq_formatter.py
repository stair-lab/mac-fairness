# script/formatter/bbq_formatter.py
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


def map_answer_to_letter(original_answer, original_choices):
    """Map original answer to letter ID (A, B, C, etc.).

    Examples:
    - "yes" -> "A" (if yes is first choice)
    - "no" -> "B" (if no is second choice)
    - 0 -> "A" (if using 0-based indexing)
    - 1 -> "B" (if using 0 or 1-based indexing)
    """
    if original_answer is None:
        return None

    # Try exact match first
    answer_str = str(original_answer).strip()
    for i, choice in enumerate(original_choices):
        if str(choice).strip() == answer_str:
            return chr(65 + i)  # Return A, B, C, etc.

    # Try as index (handle both 0-based and 1-based)
    try:
        idx = int(original_answer)
        # Check if 0-based index
        if 0 <= idx < len(original_choices):
            return chr(65 + idx)
        # Check if 1-based index
        elif 1 <= idx <= len(original_choices):
            return chr(65 + idx - 1)
    except (ValueError, TypeError):
        pass

    # Store unmapped answer in metadata for debugging
    return None


def format_bbq_to_unified(input_path, output_path, subcategory="race"):
    """Convert BBQ benchmark to unified JSONL format.

    BBQ format has:
    - Choices as separate fields: ans0, ans1, ans2
    - Correct answer indicated by 'label' field (0, 1, or 2)
    - Complex metadata like answer_info, question_polarity, etc.
    """
    with open(input_path, "r") as f_in, open(output_path, "w") as f_out:
        for idx, line in enumerate(f_in):
            if not line.strip():
                continue

            item = json.loads(line)

            # Extract choices from BBQ format (ans0, ans1, ans2)
            choices = []
            for i in range(3):  # BBQ always has 3 choices
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
                "question_id": f"bbq_{subcategory}_{item.get('example_id', idx)}",
                "source_dataset": "BBQ",
                "source_id": str(item.get("example_id", idx)),
                "question_type": "multiple_choice",
                "context": item.get("context", ""),
                "question": item["question"],
                "choices": unified_choices,  # Always A/B/C format
                "correct_answer_id": correct_answer_id,  # Always A/B/C format
                # Preserve ALL original BBQ metadata
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
                    "additional_metadata": item.get("additional_metadata"),
                },
                "schema_version": "2025-11-27",
            }

            f_out.write(json.dumps(formatted) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert BBQ benchmark to unified JSONL format"
    )
    parser.add_argument("--input", required=True, help="Path to input BBQ JSONL file")
    parser.add_argument(
        "--output", required=True, help="Path to output unified JSONL file"
    )
    parser.add_argument(
        "--subcategory",
        default="race",
        help="BBQ subcategory (e.g., race, gender, age) for question IDs",
    )

    args = parser.parse_args()
    format_bbq_to_unified(args.input, args.output, args.subcategory)
    print(f"✓ Formatted BBQ data saved to {args.output}")
