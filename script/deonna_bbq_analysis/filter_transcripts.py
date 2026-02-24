#!/usr/bin/env python3
import json
from pathlib import Path
from typing import Dict


# DRY_RUN delete: /home/groups/sanmi/project-mac-fairness-exp-root/bbq_age/bullet/rationale_first/20260127T102910.595Z_bbq-sampled-set_bullet_rationale_first_qwen3-4b_1agent_as-ai_v2025-12-10/transcript/0ac737f3-31ed-41a3-93fd-22d413828d47.json
# DRY_RUN delete: /home/groups/sanmi/project-mac-fairness-exp-root/bbq_age/bullet/rationale_first/20260127T102910.595Z_bbq-sampled-set_bullet_rationale_first_qwen3-4b_1agent_as-ai_v2025-12-10/transcript/0fd284c3-8a1e-49a5-a532-151e7fd63140.json
# DRY_RUN delete: /home/groups/sanmi/project-mac-fairness-exp-root/bbq_age/bullet/rationale_first/20260127T102910.595Z_bbq-sampled-set_bullet_rationale_first_qwen3-4b_1agent_as-ai_v2025-12-10/transcript/16ce6a17-cdd4-4d16-b410-58d2422bc85a.json
# DRY_RUN delete: /home/groups/sanmi/project-mac-fairness-exp-root/bbq_age/bullet/rationale_first/20260127T103357.753Z_bbq-sampled-set_bullet_rationale_first_qwen3-4b_1agent_as-ai_v2025-12-10/transcript/002661ab-0da3-4ff5-9d07-48b638111bf3.json
# DRY_RUN delete: /home/groups/sanmi/project-mac-fairness-exp-root/bbq_age/bullet/rationale_first/20260127T103357.753Z_bbq-sampled-set_bullet_rationale_first_qwen3-4b_1agent_as-ai_v2025-12-10/transcript/07fcca6a-169a-40e7-8eca-82717bd6425f.json
# DRY_RUN delete: /home/groups/sanmi/project-mac-fairness-exp-root/bbq_age/bullet/rationale_first/20260127T103357.753Z_bbq-sampled-set_bullet_rationale_first_qwen3-4b_1agent_as-ai_v2025-12-10/transcript/0811ae2c-4d9b-4b0e-bd95-b89f8fe9a874.json
# DRY_RUN delete: /home/groups/sanmi/project-mac-fairness-exp-root/bbq_age/bullet/rationale_first/20260127T103834.976Z_bbq-sampled-set_bullet_rationale_first_qwen3-4b_1agent_as-ai_v2025-12-10/transcript/02a98afe-960b-49d9-aa95-72224aff5e64.json
# DRY_RUN delete: /home/groups/sanmi/project-mac-fairness-exp-root/bbq_age/bullet/rationale_first/20260127T103834.976Z_bbq-sampled-set_bullet_rationale_first_qwen3-4b_1agent_as-ai_v2025-12-10/transcript/1185579d-a2f7-4808-bd96-f640f8ec8509.json
# DRY_RUN delete: /home/groups/sanmi/project-mac-fairness-exp-root/bbq_age/bullet/rationale_first/20260127T103834.976Z_bbq-sampled-set_bullet_rationale_first_qwen3-4b_1agent_as-ai_v2025-12-10/transcript/17aba660-fba4-4b0d-b346-2e3d41b3ba8d.json
# DRY_RUN delete: /home/groups/sanmi/project-mac-fairness-exp-root/bbq_age/bullet/rationale_first/20260127T104308.528Z_bbq-sampled-set_bullet_rationale_first_qwen3-4b_1agent_as-ai_v2025-12-10/transcript/063c6ba8-f290-4103-8e56-3e49d6ceafa8.json
# DRY_RUN delete: /home/groups/sanmi/project-mac-fairness-exp-root/bbq_age/bullet/rationale_first/20260127T104308.528Z_bbq-sampled-set_bullet_rationale_first_qwen3-4b_1agent_as-ai_v2025-12-10/transcript/0670fb12-04ba-4d54-9d2b-495600eabf64.json
# DRY_RUN delete: /home/groups/sanmi/project-mac-fairness-exp-root/bbq_age/bullet/rationale_first/20260127T104308.528Z_bbq-sampled-set_bullet_rationale_first_qwen3-4b_1agent_as-ai_v2025-12-10/transcript/0a0cc512-9e29-4d8d-a22a-08290403a527.json
# DRY_RUN delete: /home/groups/sanmi/project-mac-fairness-exp-root/bbq_age/bullet/rationale_first/20260127T104737.772Z_bbq-sampled-set_bullet_rationale_first_qwen3-4b_1agent_as-ai_v2025-12-10/transcript/02225075-c233-402a-87d8-f49a7133c0d8.json
# DRY_RUN delete: /home/groups/sanmi/project-mac-fairness-exp-root/bbq_age/bullet/rationale_first/20260127T104737.772Z_bbq-sampled-set_bullet_rationale_first_qwen3-4b_1agent_as-ai_v2025-12-10/transcript/03ed8cda-f42d-4ce7-9c9d-6a0566a31b39.json
# DRY_RUN delete: /home/groups/sanmi/project-mac-fairness-exp-root/bbq_age/bullet/rationale_first/20260127T104737.772Z_bbq-sampled-set_bullet_rationale_first_qwen3-4b_1agent_as-ai_v2025-12-10/transcript/04a9ac1c-893a-4468-a05e-72c4232c0bb0.json


SOCIAL_GROUPS = [
    # "bbq_age",
    "bbq_disability_status",
    # "bbq_gender_identity",
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
# add one disamig example to each transcript dir for these # "bullet", # "letter_colon", # "letter_dot", all answer_first gender_identity
DISPLAY_ORDERS = [
    # "bullet",
    # "letter_colon",
    # "letter_dot",
    # "letter_paren",
    # "arabic_colon",
    # "arabic_dot",
    # "arabic_paren",
    # "roman_colon",
    # "roman_dot",
    # "roman_paren",
    # "none",
]

JSON_FIELD_ORDERS = [
    "answer_first",
    # "rationale_first",
]

ROOT = Path("/home/groups/sanmi/project-mac-fairness-exp-root")
DATA_ROOT = Path("/scratch/users/deonnao/mac-fairness/data/BBQ")

# Configure deletion behavior
TARGET_CONTEXT = "ambig"  # set to "ambig" or "disambig"
DELETE_PER_TRANSCRIPT_DIR = 8  # set to desired number per transcript folder
DRY_RUN = True
DIAGNOSTIC = False
DIAGNOSTIC_LIMIT = 10


def load_context_map(data_file: Path) -> Dict[str, str]:
    context_map: Dict[str, str] = {}
    with data_file.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            question_id = record.get("question_id")
            context = (
                record.get("source_metadata", {})
                .get("context_condition")
            )
            if question_id is not None and context is not None:
                context_map[str(question_id)] = str(context)
    return context_map


def main() -> None:
    missing_base = 0
    missing_transcript = 0
    missing_data = 0
    missing_question_id = 0
    missing_in_data = 0
    diagnostic_shown = 0
    deleted = 0
    matched = 0

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

                    deleted_here = 0
                    for transcript_file in sorted(p for p in transcript_dir.iterdir() if p.is_file()):
                        if deleted_here >= DELETE_PER_TRANSCRIPT_DIR:
                            break

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
                            if DIAGNOSTIC and diagnostic_shown < DIAGNOSTIC_LIMIT:
                                print(f"Missing question_id in transcript: {transcript_file}")
                                diagnostic_shown += 1
                            continue

                        context = context_map.get(str(question_id))
                        if context is None:
                            missing_in_data += 1
                            if DIAGNOSTIC and diagnostic_shown < DIAGNOSTIC_LIMIT:
                                print(f"question_id not in data map: {question_id} ({transcript_file})")
                                diagnostic_shown += 1
                            continue

                        if context == TARGET_CONTEXT:
                            matched += 1
                            if DRY_RUN:
                                print(f"DRY_RUN delete: {transcript_file}")
                            else:
                                transcript_file.unlink()
                                print(f"Deleted: {transcript_file}")
                            deleted += 1
                            deleted_here += 1

    print(f"Matched context: {matched}")
    print(f"Deleted files: {deleted}")
    print(f"Missing base dirs: {missing_base}")
    print(f"Missing transcript dirs: {missing_transcript}")
    print(f"Missing data files: {missing_data}")
    print(f"Missing question_id: {missing_question_id}")
    print(f"Missing question_id in data: {missing_in_data}")


if __name__ == "__main__":
    main()
