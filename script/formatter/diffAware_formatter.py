# script/formatter/diffAware_formatter.py
import json
import argparse
import pickle
from pathlib import Path
import re

# Accept (a)/(b)/(c) OR (1)/(2)/(3) (common N3 typo), case-insensitive
CHOICE_PATTERNS = [
    # standard (a) (b) (c)
    re.compile(r"\(a\)\s*(.*?)\s*\(b\)\s*(.*?)\s*\(c\)\s*(.*)\s*$", re.IGNORECASE | re.DOTALL),
    # numeric (1) (b) (c) or (1) (2) (3)
    re.compile(r"\(1\)\s*(.*?)\s*\(b\)\s*(.*?)\s*\(c\)\s*(.*)\s*$", re.IGNORECASE | re.DOTALL),
    re.compile(r"\(1\)\s*(.*?)\s*\(2\)\s*(.*?)\s*\(3\)\s*(.*)\s*$", re.IGNORECASE | re.DOTALL),
]

def split_question_and_choices(full_text: str):
    """
    Robustly split a DiffAware question string into:
      - question stem
      - [choice_a, choice_b, choice_c]

    Handles:
      - inline: "...? (a) ... (b) ... (c) ..."
      - newline: "...?\n(a) ... (b) ... (c) ..."
      - occasional N3 typo: (1) instead of (a)
    """
    if not isinstance(full_text, str):
        return str(full_text), ["", "", ""]

    text = full_text.strip()

    # First, try to split into "stem" + "choices" using the first occurrence of "(a)" or "(1)"
    # This avoids relying on '\n', which breaks N2.
    # Find earliest marker
    a_pos = None
    for marker in ["(a)", "(A)", "(1)"]:
        p = text.find(marker)
        if p != -1:
            a_pos = p if a_pos is None else min(a_pos, p)

    if a_pos is None:
        # No recognizable markers; return whole text as question with blank choices
        return text, ["", "", ""]

    stem = text[:a_pos].strip()
    choices_blob = text[a_pos:].strip()

    # Now parse choices_blob using regex patterns
    for pat in CHOICE_PATTERNS:
        m = pat.search(choices_blob)
        if m:
            a, b, c = m.group(1).strip(), m.group(2).strip(), m.group(3).strip()
            return stem, [a, b, c]

    # If regex fails, fallback: attempt naive splits that tolerate missing whitespace/newlines.
    try:
        a = choices_blob.split("(a)", 1)[1].split("(b)", 1)[0].strip()
        b = choices_blob.split("(b)", 1)[1].split("(c)", 1)[0].strip()
        c = choices_blob.split("(c)", 1)[1].strip()
        return stem, [a, b, c]
    except Exception:
        # Try numeric fallback
        try:
            a = choices_blob.split("(1)", 1)[1].split("(b)", 1)[0].strip()
            b = choices_blob.split("(b)", 1)[1].split("(c)", 1)[0].strip()
            c = choices_blob.split("(c)", 1)[1].strip()
            return stem, [a, b, c]
        except Exception:
            return stem, ["", "", ""]


def load_diffAware_pkl_as_json_lines(input_pkl_path):
    with open(input_pkl_path, "rb") as f:
        loaded = pickle.load(f)

    records = []
    global_idx = 0

    if isinstance(loaded, (list, tuple)) and len(loaded) == 2:
        splits = [("different", loaded[0]), ("equal", loaded[1])]
    else:
        splits = [("all", loaded)]

    for split_name, split_data in splits:
        for local_idx, record in enumerate(split_data):
            if not isinstance(record, (list, tuple)) or len(record) < 2:
                continue

            question_str, label_int, example_id = record[0], record[1], record[2]

            # Cast label safely
            try:
                label_int = int(label_int)
            except Exception:
                label_int = None

            question, choices = split_question_and_choices(question_str)

            records.append(
                {
                    "global_index": global_idx,
                    "question": question,
                    "ans0": choices[0],
                    "ans1": choices[1],
                    "ans2": choices[2],
                    "label": label_int,
                    "source_id": local_idx,             
                    "example_id": example_id, 
                    "additional_metadata": {"split": split_name, "label": label_int},
                }
            )
            global_idx += 1

    return records

def format_diffAware_to_unified(input_path, output_path, subcategory="d1"):
    """
      
    """
    # Convert .pkl to JSON
    json_lines = load_diffAware_pkl_as_json_lines(input_path)

    with open(output_path, "w") as f_out:
        for item in json_lines:
            idx = item["global_index"]

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

            label = item.get("label")
            split = (item.get("additional_metadata") or {}).get("split")

            if subcategory in {"n1", "n2", "n3", "n4"} and split == "equal":
                correct_answer_id = "C" if label is not None else None
            else:
                correct_answer_id = chr(65 + label) if label is not None else None

            # Build the minimal unified format
            formatted = {
                # Unified question_id: benchmark_subcategory_originalid
                "question_id": f"diffAware_{subcategory}_{idx}",
                "source_dataset": "DiffAware",
                "source_id": str(idx),
                "question_type": "multiple_choice",
                "context": "",  # DiffAware has no separate context
                "question": item["question"],
                "choices": unified_choices,  # Always A/B/C format
                "correct_answer_id": correct_answer_id,  # Always A/B/C format
                # Preserve "metadata"; fields not in .pkl will return Null
                "source_metadata": {
                    "example_id": str(item.get("example_id")),
                    "additional_metadata": item.get("additional_metadata"),

                },
                "schema_version": "2025-12-10",
            }

            f_out.write(json.dumps(formatted) + "\n")


def format_diffaware_directory(input_dir, output_dir):
    """Process all .pkl files in input directory and save to output directory."""
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    pkl_files = sorted(input_path.glob("*.pkl"))
    if not pkl_files:
        print(f"No .pkl files found in {input_dir}")
        return

    for input_file in pkl_files:
        subcategory = input_file.stem.lower()
        output_filename = f"diffAware_{subcategory}.jsonl"
        output_file = output_path / output_filename

        print(f"Processing {input_file.name} -> {output_filename}")
        format_diffAware_to_unified(input_file, output_file, subcategory)

    print(f"Formatted {len(pkl_files)} DiffAware files to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert DiffAware .pkl benchmark to unified JSONL format"
    )
    parser.add_argument(
        "--input-dir",
        required=True,
        help="Directory containing DiffAware .pkl files (e.g., local/user/Downloads/DifferenceAware)",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output directory for formatted files (e.g., data/DifferenceAware)",
    )

    args = parser.parse_args()
    format_diffaware_directory(args.input_dir, args.output_dir)

