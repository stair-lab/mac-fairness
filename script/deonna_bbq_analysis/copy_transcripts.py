#!/usr/bin/env python3
import argparse
import shutil
from pathlib import Path

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
    "bbq_ses",
    # "bbq_sexual_orientation"
]
# cooked: gender_identity, nationality,
# use: bbq_age, bbq_physical_appearance, bbq_race_ethnicity, bbq_race_x_gender, bbq_race_x_ses, bbq_religion, bbq_sexual_orientation, disability_status, ses,
#need to fix: nationality/roman_colon/answer_first. Missing two transcripts, but not sure why. Maybe copy_transcript.py skipped them because they don't exist.
# ses needs rationale first experiments - DONE
# gender_identity needs answer first experiments, add one disamig example to each transcript dir for these # "bullet", # "letter_colon", # "letter_dot"
DISPLAY_ORDERS = [
    # "bullet",
    # "letter_colon",
    # "letter_dot",
    # "letter_paren",
    # "arabic_colon",
    # "arabic_dot",
    # "arabic_paren",
    "roman_colon",
    "roman_dot",
    "roman_paren",
    "none",
]

JSON_FIELD_ORDERS = [
    "answer_first",
    # "rationale_first",
]

SRC_ROOT = Path("/home/groups/sanmi/project-mac-fairness-exp-root/resampled")
DST_ROOT = Path("/home/groups/sanmi/project-mac-fairness-exp-root")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Copy transcript files into existing runs.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report what would be copied without writing files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dry_run = args.dry_run

    copied = 0
    skipped_existing = 0
    missing_src = 0
    missing_dst = 0

    for social_group in SOCIAL_GROUPS:
        for display_order in DISPLAY_ORDERS:
            for json_field_order in JSON_FIELD_ORDERS:
                src_base = SRC_ROOT / social_group / display_order / json_field_order
                dst_base = DST_ROOT / social_group / display_order / json_field_order

                if not src_base.is_dir():
                    missing_src += 1
                    continue
                dst_ts_dirs = sorted(p for p in dst_base.iterdir() if p.is_dir())
                if not dst_ts_dirs:
                    missing_dst += 1
                    continue

                dst_transcripts = [p / "transcript" for p in dst_ts_dirs if (p / "transcript").is_dir()]
                if not dst_transcripts:
                    missing_dst += 1
                    continue

                dest_index = 0
                for ts_dir in sorted(p for p in src_base.iterdir() if p.is_dir()):
                    src_transcript = ts_dir / "transcript"
                    if not src_transcript.is_dir():
                        continue

                    for src_file in src_transcript.iterdir():
                        if not src_file.is_file():
                            continue

                        placed = False
                        for _ in range(len(dst_transcripts)):
                            dst_transcript = dst_transcripts[dest_index]
                            dest_index = (dest_index + 1) % len(dst_transcripts)
                            dst_file = dst_transcript / src_file.name
                            if dst_file.exists():
                                skipped_existing += 1
                                continue
                            if not dry_run:
                                shutil.copy2(src_file, dst_file)
                            copied += 1
                            placed = True
                            break
                        if not placed:
                            skipped_existing += 1

    if dry_run:
        print("Dry run only; no files copied.")
    print(f"Copied files: {copied}")
    print(f"Skipped existing: {skipped_existing}")
    print(f"Missing source bases: {missing_src}")
    print(f"Missing destination bases: {missing_dst}")


if __name__ == "__main__":
    main()
