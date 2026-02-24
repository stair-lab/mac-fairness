#!/usr/bin/env python3
import json
from pathlib import Path
from typing import Dict

SOCIAL_GROUPS = [
    # "bbq_age",
    # "bbq_disability_status",
    # "bbq_gender_identity",
    # "bbq_nationality",
    # "bbq_physical_appearance",
    # "bbq_race_ethnicity",
    # "bbq_race_x_gender",
    # "bbq_race_x_ses",
    # "bbq_religion",
    # "bbq_ses",
    "bbq_sexual_orientation"
]

#need to fix: nationality/roman_colon/answer_first. Missing two transcripts, but not sure why. Maybe copy_transcript.py skipped them because they don't exist.
# disability_status needs even split, also roman_paran-rationale_first needs one more experiment
DISPLAY_ORDERS = [
    "bullet",
    "letter_colon",
    "letter_dot",
    "letter_paren",
    "arabic_colon",
    "arabic_dot",
    "arabic_paren",
    "roman_colon",
    "roman_dot",
    "roman_paren",
    "none",
]

JSON_FIELD_ORDERS = [
    "answer_first",
    "rationale_first",
]

ROOT = Path("/home/groups/sanmi/project-mac-fairness-exp-root")
DATA_ROOT = Path("/scratch/users/deonnao/mac-fairness/data/BBQ")


def load_context_map(data_file: Path) -> Dict[str, str]:
    context_map: Dict[str, str] = {}
    with data_file.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            question_id = record.get("question_id")
            context = record.get("source_metadata", {}).get("context_condition")
            if question_id is not None and context is not None:
                context_map[str(question_id)] = str(context)
    return context_map


def main() -> None:
    missing_base = 0
    missing_transcript = 0
    missing_data = 0
    missing_question_id = 0
    missing_in_data = 0

    context_cache: Dict[str, Dict[str, str]] = {}

    for social_group in SOCIAL_GROUPS:
        data_file = DATA_ROOT / f"{social_group}.jsonl"
        if not data_file.is_file():
            missing_data += 1
            continue
        if social_group not in context_cache:
            context_cache[social_group] = load_context_map(data_file)
        context_map = context_cache[social_group]

        for display_order in DISPLAY_ORDERS:
            for json_field_order in JSON_FIELD_ORDERS:
                base = ROOT / social_group / display_order / json_field_order
                if not base.is_dir():
                    missing_base += 1
                    continue

                ts_dirs = sorted(p for p in base.iterdir() if p.is_dir())
                if not ts_dirs:
                    missing_base += 1
                    continue

                for ts_dir in ts_dirs:
                    transcript_dir = ts_dir / "transcript"
                    if not transcript_dir.is_dir():
                        missing_transcript += 1
                        continue

                    ambig = 0
                    disambig = 0
                    for transcript_file in sorted(p for p in transcript_dir.iterdir() if p.is_file()):
                        try:
                            with transcript_file.open("r", encoding="utf-8") as handle:
                                transcript = json.load(handle)
                        except json.JSONDecodeError:
                            continue

                        question_id = (
                            transcript.get("question_id")
                            or transcript.get("experiment_metadata", {}).get("question_id")
                        )
                        if question_id is None:
                            missing_question_id += 1
                            continue

                        context = context_map.get(str(question_id))
                        if context is None:
                            missing_in_data += 1
                            continue

                        if context == "ambig":
                            ambig += 1
                        elif context == "disambig":
                            disambig += 1

                    print(f"{transcript_dir} ambig={ambig} disambig={disambig}")

    print(f"Missing base dirs: {missing_base}")
    print(f"Missing transcript dirs: {missing_transcript}")
    print(f"Missing data files: {missing_data}")
    print(f"Missing question_id: {missing_question_id}")
    print(f"Missing question_id in data: {missing_in_data}")


if __name__ == "__main__":
    main()


