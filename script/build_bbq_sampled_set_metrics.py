# #!/usr/bin/env python3
# import argparse
# import csv
# import json
# import sys
# from pathlib import Path
# from typing import Dict, Iterable, List, Optional, Tuple


# SUBCATEGORIES = [
#     "bbq_age",
#     # "bbq_disability_status",
#     # "bbq_gender_identity",
#     # "bbq_nationality",
#     # "bbq_physical_appearance",
#     # "bbq_race_ethnicity",
#     # "bbq_race_x_gender",
#     # "bbq_race_x_ses",
#     # "bbq_religion",
#     # "bbq_ses",
#     # "bbq_sexual_orientation",
# ]

# DISPLAY_ORDERS = [
#     "bullet",
#     # "letter_colon",
#     # "letter_dot",
#     # "letter_paren",
#     # "arabic_colon",
#     # "arabic_dot",
#     # "arabic_paren",
#     # "roman_colon",
#     # "roman_dot",
#     # "roman_paren",
#     # "none",
# ]

# JSON_FIELD_ORDERS = [
#     "answer_first",
#     # "rationale_first",
# ]

# UNKNOWNS = {
#     "unknown",
#     "cannot be determined",
#     "can't be determined",
#     "not answerable",
#     "not known",
#     "not enough info",
#     "not enough information",
#     "cannot answer",
#     "can't answer",
#     "undetermined",
# }


# def extract_bias_target_and_unknown_index(item: dict) -> Tuple[Optional[int], Optional[int]]:
#     sm = item.get("source_metadata")
#     if not isinstance(sm, dict):
#         return None, None
#     answer_info = sm.get("answer_info")
#     if not isinstance(answer_info, dict):
#         return None, None
#     additional = sm.get("additional_metadata")
#     stereotyped = set()
#     if isinstance(additional, dict):
#         sg = additional.get("stereotyped_groups")
#         if isinstance(sg, list):
#             stereotyped = {str(x).lower() for x in sg if x is not None}

#     bias_target_idx = None
#     unknown_idx = None

#     unknowns_l = {u.lower() for u in UNKNOWNS}
#     for k, tags in answer_info.items():
#         if not isinstance(tags, list):
#             continue
#         tags_l = {str(t).lower() for t in tags if t is not None}
#         if tags_l & unknowns_l:
#             try:
#                 unknown_idx = int(str(k).replace("ans", ""))
#             except ValueError:
#                 pass
#         if stereotyped:
#             hit = False
#             for tag in tags_l:
#                 for st in stereotyped:
#                     if st and st in tag:
#                         hit = True
#                         break
#                 if hit:
#                     break
#             if hit:
#                 try:
#                     bias_target_idx = int(str(k).replace("ans", ""))
#                 except ValueError:
#                     pass

#     return bias_target_idx, unknown_idx


# def extract_question_id(payload: dict) -> Optional[str]:
#     if "question_id" in payload:
#         return payload.get("question_id")
#     meta = payload.get("experiment_metadata")
#     if isinstance(meta, dict):
#         return meta.get("question_id")
#     return None


# def extract_opinion(payload: dict) -> Optional[str]:
#     rounds = payload.get("conversation_rounds")
#     if not isinstance(rounds, list):
#         return None
#     for r in reversed(rounds):
#         if not isinstance(r, dict):
#             continue
#         messages = r.get("messages")
#         if not isinstance(messages, list):
#             continue
#         for m in reversed(messages):
#             if not isinstance(m, dict):
#                 continue
#             structured = m.get("structured_response")
#             if isinstance(structured, dict) and "opinion" in structured:
#                 return structured.get("opinion")
#     return None


# def load_correct_answer_map(
#     jsonl_path: Path,
# ) -> Dict[str, Tuple[str, Optional[str], Optional[int], Optional[int]]]:
#     mapping: Dict[str, Tuple[str, Optional[str], Optional[int], Optional[int]]] = {}
#     if not jsonl_path.exists():
#         return mapping
#     with jsonl_path.open("r", encoding="utf-8") as f:
#         for line in f:
#             line = line.strip()
#             if not line:
#                 continue
#             try:
#                 row = json.loads(line)
#             except json.JSONDecodeError:
#                 continue
#             qid = row.get("question_id")
#             ca = row.get("correct_answer_id")
#             source_meta = row.get("source_metadata")
#             context_condition = None
#             if isinstance(source_meta, dict):
#                 context_condition = source_meta.get("context_condition")
#             bias_target_idx, unknown_idx = extract_bias_target_and_unknown_index(row)
#             if qid is None or ca is None:
#                 continue
#             if not isinstance(qid, str):
#                 qid = str(qid)
#             if not isinstance(ca, str):
#                 ca = str(ca)
#             if context_condition is not None and not isinstance(context_condition, str):
#                 context_condition = str(context_condition)
#             mapping[qid] = (ca, context_condition, bias_target_idx, unknown_idx)
#     return mapping


# def iter_transcript_jsons(root: Path) -> Iterable[Path]:
#     for transcript_dir in root.rglob("transcript"):
#         if not transcript_dir.is_dir():
#             continue
#         for json_path in transcript_dir.glob("*.json"):
#             yield json_path

# """
# Get the folder structure from experiment root to the transcript json to use as grouping key to create 
# a similar folder structure for later.

# Returns a tuple of (subcategory, display_order, json_field_order, some_folder) if valid, else None.
# """
# def group_key(root: Path, json_path: Path) -> Optional[Tuple[str, str, str, str]]:
#     try:
#         rel = json_path.relative_to(root).parts
#     except ValueError:
#         return None
#     if len(rel) < 5 or rel[-2] != "transcript":
#         return None
#     # rel: {subcategory}/{display}/{json_field}/{some_folder}/transcript/{file}.json
#     return rel[0], rel[1], rel[2], rel[3]


# def main() -> int:
#     parser = argparse.ArgumentParser(
#         description="Extract opinions and join with BBQ correct answers."
#     )
#     parser.add_argument(
#         "--transcript-root",
#         default="/home/groups/sanmi/project-mac-fairness-exp-root",
#         help="Root containing subcategory/display/json_field_order/.../transcript/*.json",
#     )
#     parser.add_argument(
#         "--bbq-root",
#         default="/scratch/users/deonnao/mac-fairness/data/bbq",
#         help="Root containing {subcategory}.jsonl files",
#     )
#     parser.add_argument(
#         "--output-root",
#         default="/scratch/users/deonnao/mac-fairness/metric_data",
#         help="Root to write CSVs under {subcategory}/{display}/{json_field_order}/",
#     )
#     args = parser.parse_args()

#     transcript_root = Path(args.transcript_root)
#     bbq_root = Path(args.bbq_root)
#     output_root = Path(args.output_root)

#     grouped: Dict[Tuple[str, str, str, str], Dict[str, str]] = {}

#     for json_path in iter_transcript_jsons(transcript_root):
#         key = group_key(transcript_root, json_path)
#         if key is None:
#             continue
#         subcategory, display_order, json_field_order, _some_folder = key
#         if (
#             subcategory not in SUBCATEGORIES
#             or display_order not in DISPLAY_ORDERS
#             or json_field_order not in JSON_FIELD_ORDERS
#         ):
#             continue
#         try:
#             payload = json.loads(json_path.read_text(encoding="utf-8"))
#         except (json.JSONDecodeError, OSError):
#             continue
#         qid = extract_question_id(payload)
#         opinion = extract_opinion(payload)
#         if not qid or not opinion:
#             continue
#         grouped.setdefault(key, {})[qid] = opinion

#     correct_answer_cache: Dict[
#         str, Dict[str, Tuple[str, Optional[str], Optional[int], Optional[int]]]
#     ] = {}

#     for (subcategory, display_order, json_field_order, some_folder), rows in grouped.items():
#         if subcategory not in correct_answer_cache:
#             jsonl_path = bbq_root / f"{subcategory}.jsonl"
#             if not jsonl_path.exists():
#                 alt = bbq_root.parent / "BBQ" / f"{subcategory}.jsonl"
#                 if alt.exists():
#                     jsonl_path = alt
#             correct_answer_cache[subcategory] = load_correct_answer_map(jsonl_path)
#             if not correct_answer_cache[subcategory]:
#                 print(
#                     f"[warn] No correct_answer_id loaded for {subcategory} "
#                     f"from {jsonl_path}",
#                     file=sys.stderr,
#                 )
#         answer_map = correct_answer_cache[subcategory]

#         out_dir = output_root / subcategory / display_order / json_field_order / some_folder
#         out_dir.mkdir(parents=True, exist_ok=True)
#         if args.output_name:
#             print(
#                 "[warn] --output-name is deprecated and ignored; "
#                 "using derived filename instead.",
#                 file=sys.stderr,
#             )
#         out_path = out_dir / f"{subcategory}_{display_order}_{json_field_order}.csv"

#         rows_sorted = sorted(rows.items(), key=lambda r: r[0])
#         with out_path.open("w", encoding="utf-8", newline="") as f:
#             writer = csv.writer(f)
#             writer.writerow(
#                 [
#                     "question_id",
#                     "opinion",
#                     "correct_answer_id",
#                     "context_condition",
#                     "bias_target",
#                     "unknown",
#                 ]
#             )
#             for qid, opinion in rows_sorted:
#                 ca, context, bias_target, unknown = (None, None, None, None)
#                 if qid in answer_map:
#                     ca, context, bias_target, unknown = answer_map[qid]
#                 writer.writerow([qid, opinion, ca, context, bias_target, unknown])

#     return 0


# if __name__ == "__main__":
#     raise SystemExit(main())


#!/usr/bin/env python3
import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


SUBCATEGORIES = [
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
    "bbq_sexual_orientation",
]

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
    # "answer_first",
    "rationale_first",
]

UNKNOWNS = {
    "unknown",
    "cannot be determined",
    "can't be determined",
    "not answerable",
    "not known",
    "not enough info",
    "not enough information",
    "cannot answer",
    "can't answer",
    "undetermined",
}


def extract_bias_target_and_unknown_index(item: dict) -> Tuple[Optional[int], Optional[int]]:
    sm = item.get("source_metadata")
    if not isinstance(sm, dict):
        return None, None
    answer_info = sm.get("answer_info")
    if not isinstance(answer_info, dict):
        return None, None
    additional = sm.get("additional_metadata")
    stereotyped = set()
    if isinstance(additional, dict):
        sg = additional.get("stereotyped_groups")
        if isinstance(sg, list):
            stereotyped = {str(x).lower() for x in sg if x is not None}

    bias_target_idx = None
    unknown_idx = None

    unknowns_l = {u.lower() for u in UNKNOWNS}
    for k, tags in answer_info.items():
        if not isinstance(tags, list):
            continue
        tags_l = {str(t).lower() for t in tags if t is not None}
        if tags_l & unknowns_l:
            try:
                unknown_idx = int(str(k).replace("ans", ""))
            except ValueError:
                pass
        if stereotyped:
            hit = False
            for tag in tags_l:
                for st in stereotyped:
                    if st and st in tag:
                        hit = True
                        break
                if hit:
                    break
            if hit:
                try:
                    bias_target_idx = int(str(k).replace("ans", ""))
                except ValueError:
                    pass

    return bias_target_idx, unknown_idx


def extract_question_id(payload: dict) -> Optional[str]:
    if "question_id" in payload:
        return payload.get("question_id")
    meta = payload.get("experiment_metadata")
    if isinstance(meta, dict):
        return meta.get("question_id")
    return None


def extract_opinion(payload: dict) -> Optional[str]:
    rounds = payload.get("conversation_rounds")
    if not isinstance(rounds, list):
        return None
    for r in reversed(rounds):
        if not isinstance(r, dict):
            continue
        messages = r.get("messages")
        if not isinstance(messages, list):
            continue
        for m in reversed(messages):
            if not isinstance(m, dict):
                continue
            structured = m.get("structured_response")
            if isinstance(structured, dict) and "opinion" in structured:
                return structured.get("opinion")
    return None


def load_correct_answer_map(
    jsonl_path: Path,
) -> Dict[str, Tuple[str, Optional[str], Optional[int], Optional[int]]]:
    mapping: Dict[str, Tuple[str, Optional[str], Optional[int], Optional[int]]] = {}
    if not jsonl_path.exists():
        return mapping
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            qid = row.get("question_id")
            ca = row.get("correct_answer_id")
            source_meta = row.get("source_metadata")
            context_condition = None
            if isinstance(source_meta, dict):
                context_condition = source_meta.get("context_condition")
            bias_target_idx, unknown_idx = extract_bias_target_and_unknown_index(row)
            if qid is None or ca is None:
                continue
            if not isinstance(qid, str):
                qid = str(qid)
            if not isinstance(ca, str):
                ca = str(ca)
            if context_condition is not None and not isinstance(context_condition, str):
                context_condition = str(context_condition)
            mapping[qid] = (ca, context_condition, bias_target_idx, unknown_idx)
    return mapping


def iter_transcript_jsons(root: Path) -> Iterable[Path]:
    for transcript_dir in root.rglob("transcript"):
        if not transcript_dir.is_dir():
            continue
        for json_path in transcript_dir.glob("*.json"):
            yield json_path

"""
Get the folder structure from experiment root to the transcript json to use as grouping key to create 
a similar folder structure for later.

Returns a tuple of (subcategory, display_order, json_field_order, some_folder) if valid, else None.
"""
def group_key(root: Path, json_path: Path) -> Optional[Tuple[str, str, str, str]]:
    try:
        rel = json_path.relative_to(root).parts
    except ValueError:
        return None
    if len(rel) < 5 or rel[-2] != "transcript":
        return None
    # rel: {subcategory}/{display}/{json_field}/{some_folder}/transcript/{file}.json
    return rel[0], rel[1], rel[2], rel[3]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Extract opinions and join with BBQ correct answers."
    )
    parser.add_argument(
        "--transcript-root",
        default="/home/groups/sanmi/project-mac-fairness-exp-root",
        help="Root containing subcategory/display/json_field_order/.../transcript/*.json",
    )
    parser.add_argument(
        "--bbq-root",
        default="/scratch/users/deonnao/mac-fairness/data/bbq",
        help="Root containing {subcategory}.jsonl files",
    )
    parser.add_argument(
        "--output-root",
        default="/scratch/users/deonnao/mac-fairness/metric_data",
        help="Root to write CSVs under {subcategory}/{display}/{json_field_order}/",
    )
    parser.add_argument(
        "--output-name",
        default=None,
        help="Deprecated and ignored. Output filename is always derived from traversal.",
    )
    args = parser.parse_args()

    transcript_root = Path(args.transcript_root)
    bbq_root = Path(args.bbq_root)
    output_root = Path(args.output_root)

    grouped: Dict[Tuple[str, str, str, str], Dict[str, str]] = {}

    for json_path in iter_transcript_jsons(transcript_root):
        key = group_key(transcript_root, json_path)
        if key is None:
            continue
        subcategory, display_order, json_field_order, _some_folder = key
        if (
            subcategory not in SUBCATEGORIES
            or display_order not in DISPLAY_ORDERS
            or json_field_order not in JSON_FIELD_ORDERS
        ):
            continue
        try:
            payload = json.loads(json_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        qid = extract_question_id(payload)
        opinion = extract_opinion(payload)
        if not qid or not opinion:
            continue
        grouped.setdefault(key, {})[qid] = opinion

    correct_answer_cache: Dict[
        str, Dict[str, Tuple[str, Optional[str], Optional[int], Optional[int]]]
    ] = {}

    for (subcategory, display_order, json_field_order, some_folder), rows in grouped.items():
        if subcategory not in correct_answer_cache:
            jsonl_path = bbq_root / f"{subcategory}.jsonl"
            if not jsonl_path.exists():
                alt = bbq_root.parent / "BBQ" / f"{subcategory}.jsonl"
                if alt.exists():
                    jsonl_path = alt
            correct_answer_cache[subcategory] = load_correct_answer_map(jsonl_path)
            if not correct_answer_cache[subcategory]:
                print(
                    f"[warn] No correct_answer_id loaded for {subcategory} "
                    f"from {jsonl_path}",
                    file=sys.stderr,
                )
        answer_map = correct_answer_cache[subcategory]

        out_dir = output_root / subcategory / display_order / json_field_order / some_folder
        out_dir.mkdir(parents=True, exist_ok=True)
        if args.output_name:
            print(
                "[warn] --output-name is deprecated and ignored; "
                "using derived filename instead.",
                file=sys.stderr,
            )
        out_path = out_dir / f"{subcategory}_{display_order}_{json_field_order}.csv"

        rows_sorted = sorted(rows.items(), key=lambda r: r[0])
        with out_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "question_id",
                    "opinion",
                    "correct_answer_id",
                    "context_condition",
                    "bias_target",
                    "unknown",
                ]
            )
            for qid, opinion in rows_sorted:
                ca, context, bias_target, unknown = (None, None, None, None)
                if qid in answer_map:
                    ca, context, bias_target, unknown = answer_map[qid]
                writer.writerow([qid, opinion, ca, context, bias_target, unknown])

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
