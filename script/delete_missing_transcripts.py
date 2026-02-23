#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Iterable, Optional, Set


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Delete transcript files whose question_id is in a provided list."
    )
    parser.add_argument(
        "--ids-file",
        type=Path,
        required=True,
        help="Text file with question_ids, one per line.",
    )
    parser.add_argument(
        "--root",
        type=Path,
        required=True,
        help="Root directory that contains display_order/json_field_order run folders.",
    )
    parser.add_argument(
        "--canonical",
        type=Path,
        default=None,
        help="Canonical transcript folder to skip deletions in.",
    )
    parser.add_argument(
        "--pattern",
        default="*.json",
        help="Glob pattern for transcript files (default: *.json).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report deletions without removing files.",
    )
    return parser.parse_args()


def load_ids(ids_file: Path) -> Set[str]:
    ids: Set[str] = set()
    with ids_file.open("r", encoding="utf-8") as handle:
        for line in handle:
            qid = line.strip()
            if qid:
                ids.add(qid)
    return ids


def iter_transcript_files(root: Path, pattern: str) -> Iterable[Path]:
    yield from root.rglob(pattern)


def extract_question_id(transcript_path: Path) -> Optional[str]:
    try:
        with transcript_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None
    question_id = data.get("question_id")
    if question_id is None:
        question_id = data.get("experiment_metadata", {}).get("question_id")
    if question_id is None:
        return None
    return str(question_id)


def main() -> None:
    args = parse_args()
    ids = load_ids(args.ids_file)
    if not ids:
        raise SystemExit("No question_ids found in ids file.")

    canonical_resolved = args.canonical.resolve() if args.canonical else None
    deleted = 0
    matched = 0
    matched_ids: Set[str] = set()
    missing_id = 0

    print(f"Unique question_ids in file: {len(ids)}")

    for transcript_path in iter_transcript_files(args.root, args.pattern):
        if not transcript_path.is_file():
            continue
        if canonical_resolved and canonical_resolved in transcript_path.resolve().parents:
            continue
        question_id = extract_question_id(transcript_path)
        if question_id is None:
            missing_id += 1
            continue
        if question_id in ids:
            matched += 1
            matched_ids.add(question_id)
            if args.dry_run:
                print(f"DRY_RUN delete: {transcript_path}")
            else:
                transcript_path.unlink()
                print(f"Deleted: {transcript_path}")
            deleted += 1

    if args.dry_run:
        print("Dry run only; no files deleted.")
    print(f"Matched question_ids: {matched}")
    print(f"Matched unique question_ids: {len(matched_ids)}")
    print(f"Deleted files: {deleted}")
    print(f"Missing question_id in files: {missing_id}")


if __name__ == "__main__":
    main()
