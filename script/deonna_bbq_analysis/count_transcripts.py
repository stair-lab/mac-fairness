#!/usr/bin/env python3
from pathlib import Path

SOCIAL_GROUPS = [
    # "bbq_age",
    # "bbq_disability_status",
    "bbq_gender_identity",
    # "bbq_nationality",
    # "bbq_physical_appearance",
    # "bbq_race_ethnicity",
    # "bbq_race_x_gender",
    # "bbq_race_x_ses",
    # "bbq_religion",
    # "bbq_ses",
    # "bbq_sexual_orientation"
]

#need to fix: nationality/roman_colon/answer_first. Missing two transcripts, but not sure why. Maybe copy_transcript.py skipped them because they don't exist.

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
    # "rationale_first",
]

ROOT = Path("/home/groups/sanmi/project-mac-fairness-exp-root")


def count_files(transcript_dir: Path) -> int:
    return sum(
        1
        for p in transcript_dir.iterdir()
        if p.is_file() and p.suffix == ".json" and not p.name.startswith(".")
    )


def main() -> None:
    missing_base = 0
    missing_transcript = 0

    for social_group in SOCIAL_GROUPS:
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
                    total = count_files(transcript_dir)
                    print(f"{transcript_dir} {total}")

    print(f"Missing base dirs: {missing_base}")
    print(f"Missing transcript dirs: {missing_transcript}")


if __name__ == "__main__":
    main()
