#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare transcript directories by question_id and list files missing in the other directory."
        )
    )
    parser.add_argument(
        "--scan-root",
        type=Path,
        default=None,
        help="Root directory to scan display_order/json_field_order runs.",
    )
    parser.add_argument(
        "--canonical",
        type=Path,
        default=None,
        help="Canonical transcript directory to compare against when using --scan-root.",
    )
    parser.add_argument(
        "--missing-union-out",
        type=Path,
        default=None,
        help="Write unique missing question_ids (scan mode) to this file, one per line.",
    )
    parser.add_argument(
        "left_dir",
        type=Path,
        nargs="?",
        help="Directory containing transcripts to compare from.",
    )
    parser.add_argument(
        "right_dir",
        type=Path,
        nargs="*",
        help="One or more directories to compare against.",
    )
    parser.add_argument(
        "--pattern",
        default="*.json",
        help="Glob pattern for transcript files (default: *.json).",
    )
    parser.add_argument(
        "--no-recursive",
        action="store_true",
        help="Only search the top-level directory (no recursive scan).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional CSV output path for missing-in-right transcripts (single right dir only).",
    )
    parser.add_argument(
        "--output-missing-left",
        type=Path,
        default=None,
        help="Optional CSV output for transcripts missing in left (single right dir only).",
    )
    parser.add_argument(
        "--relative-to",
        type=Path,
        default=None,
        help="If set, store transcript paths relative to this directory in outputs.",
    )
    return parser.parse_args()


def iter_transcript_files(root: Path, pattern: str, recursive: bool) -> Iterable[Path]:
    if recursive:
        yield from root.rglob(pattern)
    else:
        yield from root.glob(pattern)


def extract_question_id(transcript_path: Path) -> Tuple[Optional[str], Optional[str]]:
    try:
        with transcript_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except json.JSONDecodeError:
        return None, "invalid_json"
    except OSError:
        return None, "read_error"

    question_id = data.get("question_id")
    if question_id is None:
        question_id = data.get("experiment_metadata", {}).get("question_id")
    if question_id is None:
        return None, "missing_question_id"
    return str(question_id), None


def index_transcripts(
    root: Path, pattern: str, recursive: bool
) -> Tuple[Dict[str, List[Path]], Dict[str, int], int]:
    index: Dict[str, List[Path]] = {}
    issues = {"invalid_json": 0, "read_error": 0, "missing_question_id": 0}
    total_files = 0
    for transcript_path in iter_transcript_files(root, pattern, recursive):
        if not transcript_path.is_file():
            continue
        total_files += 1
        question_id, issue = extract_question_id(transcript_path)
        if question_id is None:
            if issue in issues:
                issues[issue] += 1
            continue
        index.setdefault(question_id, []).append(transcript_path)
    return index, issues, total_files


def maybe_relative(path: Path, base: Optional[Path]) -> str:
    if base is None:
        return str(path)
    try:
        return str(path.relative_to(base))
    except ValueError:
        return str(path)


def write_missing_csv(
    output_path: Path,
    missing_ids: Iterable[str],
    source_index: Dict[str, List[Path]],
    target_index: Dict[str, List[Path]],
    relative_to: Optional[Path],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["question_id", "transcript_file", "count_in_source", "count_in_target"])
        for question_id in sorted(missing_ids):
            files = source_index.get(question_id, [])
            transcript_file = maybe_relative(files[0], relative_to) if files else ""
            writer.writerow([question_id, transcript_file, len(files), len(target_index.get(question_id, []))])


def main() -> None:
    args = parse_args()
    recursive = not args.no_recursive

    if args.scan_root:
        if args.canonical is None:
            raise SystemExit("--scan-root requires --canonical")
        canonical_index, canonical_issues, canonical_total = index_transcripts(
            args.canonical, args.pattern, recursive
        )
        print(f"Canonical total files: {canonical_total}")
        print(f"Canonical unique question_ids: {len(canonical_index)}")
        print(f"Canonical issues: {canonical_issues}")

        scan_root = args.scan_root
        canonical_resolved = args.canonical.resolve()
        display_dirs = sorted(p for p in scan_root.iterdir() if p.is_dir())
        missing_union: set[str] = set()
        for display_dir in display_dirs:
            field_dirs = sorted(p for p in display_dir.iterdir() if p.is_dir())
            for field_dir in field_dirs:
                run_dirs = sorted(p for p in field_dir.iterdir() if p.is_dir())
                missing_total = 0
                folders_counted = 0
                for run_dir in run_dirs:
                    transcript_dir = run_dir / "transcript"
                    if not transcript_dir.is_dir():
                        continue
                    if transcript_dir.resolve() == canonical_resolved:
                        continue
                    run_index, _, _ = index_transcripts(transcript_dir, args.pattern, recursive)
                    missing = [qid for qid in canonical_index.keys() if qid not in run_index]
                    missing_total += len(missing)
                    missing_union.update(missing)
                    folders_counted += 1
                print(
                    f"{display_dir.name}\t{field_dir.name}\tmissing_total={missing_total}\tfolders={folders_counted}"
                )
        print(f"Total missing unique question_ids: {len(missing_union)}")
        if args.missing_union_out:
            args.missing_union_out.parent.mkdir(parents=True, exist_ok=True)
            with args.missing_union_out.open("w", encoding="utf-8") as handle:
                for question_id in sorted(missing_union):
                    handle.write(f"{question_id}\n")
            print(f"Wrote missing IDs: {args.missing_union_out}")
        return

    if args.left_dir is None or not args.right_dir:
        raise SystemExit("left_dir and right_dir are required unless --scan-root is used")

    left_index, left_issues, left_total = index_transcripts(
        args.left_dir, args.pattern, recursive
    )
    print(f"Left total files: {left_total}")
    print(f"Left unique question_ids: {len(left_index)}")
    print(f"Left issues: {left_issues}")

    left_dupes = {qid: files for qid, files in left_index.items() if len(files) > 1}
    if left_dupes:
        print(f"Left duplicate question_ids: {len(left_dupes)}")
        for qid in sorted(left_dupes):
            print(f"LEFT_DUP\t{qid}\t{len(left_dupes[qid])}")

    total_missing_overall = 0
    for right_dir in args.right_dir:
        right_index, right_issues, right_total = index_transcripts(
            right_dir, args.pattern, recursive
        )
        missing_in_right = [qid for qid in left_index.keys() if qid not in right_index]
        total_missing_overall += len(missing_in_right)

        print(f"Right dir: {right_dir}")
        print(f"Right total files: {right_total}")
        print(f"Right unique question_ids: {len(right_index)}")
        print(f"Missing in right: {len(missing_in_right)}")
        print(f"Right issues: {right_issues}")

        right_dupes = {qid: files for qid, files in right_index.items() if len(files) > 1}
        if right_dupes:
            print(f"Right duplicate question_ids: {len(right_dupes)}")
            for qid in sorted(right_dupes):
                print(f"RIGHT_DUP\t{qid}\t{len(right_dupes[qid])}")

        for question_id in sorted(missing_in_right):
            files = left_index.get(question_id, [])
            transcript_file = maybe_relative(files[0], args.relative_to) if files else ""
            print(f"MISSING\t{right_dir}\t{question_id}\t{transcript_file}")

        if args.output or args.output_missing_left:
            if len(args.right_dir) > 1:
                raise SystemExit("--output and --output-missing-left require a single right_dir")
            if args.output:
                write_missing_csv(
                    args.output,
                    missing_in_right,
                    left_index,
                    right_index,
                    args.relative_to,
                )
            if args.output_missing_left:
                missing_in_left = [qid for qid in right_index.keys() if qid not in left_index]
                write_missing_csv(
                    args.output_missing_left,
                    missing_in_left,
                    right_index,
                    left_index,
                    args.relative_to,
                )
            if args.output:
                print(f"Wrote: {args.output}")

    if len(args.right_dir) > 1:
        print(f"Total missing across right dirs: {total_missing_overall}")


if __name__ == "__main__":
    main()
