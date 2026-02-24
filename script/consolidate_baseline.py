"""
Consolidate 1-agent baseline transcripts into human-readable JSONL files.

Output: experiment/consolidated_baseline/{subcategory}.jsonl

Structure mirrors consolidate_2agent.py with identity fields set to null
and single-agent columns instead of identity_agent/vanilla_agent pairs.
"""
import json
import os
import re
from functools import lru_cache
from pathlib import Path

import yaml

WORKSPACE = Path(os.environ.get("MAC_FAIRNESS_WORKSPACE", "."))
EXP_ROOT = Path(os.environ.get("MAC_FAIRNESS_EXPERIMENT_ROOT", WORKSPACE / "experiment"))
OUT_DIR = EXP_ROOT / "consolidated_baseline"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PAT = re.compile(r"\d{8}T\d{6}\.\d{3}Z_(.+?)_1agent_as-ai")


def load_correct_answers(workspace: Path) -> dict[str, str]:
    """Build question_id → correct_answer_id lookup from all benchmark JSONL files."""
    lookup = {}
    for jsonl in workspace.glob("data/**/*.jsonl"):
        try:
            for line in jsonl.read_text().splitlines():
                if not line.strip():
                    continue
                q = json.loads(line)
                qid = q.get("question_id")
                ans = q.get("correct_answer_id")
                if qid and ans:
                    lookup[qid] = ans
        except Exception:
            pass
    return lookup


@lru_cache(maxsize=2048)
def load_config_snapshot(snap_path: str) -> dict:
    """Cache config snapshot YAML reads."""
    resolved = snap_path.replace("$MAC_FAIRNESS_WORKSPACE", str(WORKSPACE))
    try:
        return yaml.safe_load(Path(resolved).read_text()) or {}
    except Exception:
        return {}


def extract_config_fields(snap: dict) -> dict:
    """Extract choice_display_format and json_field_order from config snapshot."""
    prompt_cfg = snap.get("prompt_template_config", {}).get("for_participant", {})
    return {
        "choice_display_format": prompt_cfg.get("choice_display_format"),
        "json_field_order": prompt_cfg.get("json_field_order"),
    }


def extract_row(transcript_path: Path, correct_answers: dict) -> dict | None:
    try:
        data = json.loads(transcript_path.read_bytes())
    except Exception:
        return None

    meta = data.get("experiment_metadata", {})
    summary = data.get("conversation_summary", {})
    experiment_name = meta.get("experiment_name", "")
    question_id = meta.get("question_id")

    m = MODEL_PAT.search(experiment_name)
    model = m.group(1) if m else "unknown"

    # Load config snapshot for format fields
    snap_path = meta.get("config_snapshot_path", "")
    snap = load_config_snapshot(snap_path)
    config_fields = extract_config_fields(snap)

    # Extract first-round answer + rationale
    rounds = data.get("conversation_rounds", [])
    opinion = rationale = None
    if rounds:
        msgs = rounds[0].get("messages", [])
        if msgs:
            sr = msgs[0].get("structured_response", {})
            opinion = sr.get("opinion")
            rationale = sr.get("rationale")

    final_answers = summary.get("final_answers", {})
    final_answer = next(iter(final_answers.values()), None) if final_answers else None
    correct = correct_answers.get(question_id)
    status = summary.get("status")

    # Derived fields
    is_correct = (final_answer == correct) if (final_answer and correct) else None

    # Human-readable summary
    fmt = config_fields["choice_display_format"] or "?"
    order = config_fields["json_field_order"] or "?"
    summary_text = (
        f"[{model} | {fmt} | {order}] "
        f"answer:{final_answer or '?'} correct:{correct or '?'} "
        f"is_correct:{is_correct}"
    )

    return {
        # Identifiers
        "transcript_id": data.get("transcript_id"),
        "question_id": question_id,
        "model": model,
        "benchmark_subcategory": meta.get("benchmark_subcategory"),
        "experiment_name": experiment_name,
        # Identity condition (null for baseline — single agent, no persona/demographics)
        "persona": None,
        "demographics": None,
        "if_as_human": None,
        "reveal_condition": None,
        # Config sweep parameters
        "choice_display_format": config_fields["choice_display_format"],
        "json_field_order": config_fields["json_field_order"],
        # Conversation outcome
        "status": status,
        "total_rounds": summary.get("total_rounds"),
        "correct_answer": correct,
        "final_answer": final_answer,
        "is_correct": is_correct,
        # Agent response
        "opinion": opinion,
        "rationale": rationale,
        # Human-readable
        "summary": summary_text,
        "created_at": data.get("created_at"),
        # Full transcript for qualitative review
        "conversation_rounds": rounds,
    }


def main():
    print("Loading correct answers from benchmark data...", flush=True)
    correct_answers = load_correct_answers(WORKSPACE)
    print(f"  Loaded {len(correct_answers)} question→answer mappings", flush=True)

    subcats = sorted([d.name for d in EXP_ROOT.iterdir() if d.is_dir() and not d.name.startswith("consolidated")])
    print(f"Found {len(subcats)} subcategories")

    total = 0
    for subcat in subcats:
        subcat_dir = EXP_ROOT / subcat
        # Only 1-agent baseline directories
        exp_dirs = [d for d in subcat_dir.iterdir()
                    if d.is_dir() and "1agent_as-ai" in d.name]
        if not exp_dirs:
            continue
        print(f"\n  {subcat}: {len(exp_dirs)} experiment dirs", flush=True)

        out_path = OUT_DIR / f"{subcat}.jsonl"
        count = 0
        with out_path.open("w") as f:
            for exp_dir in exp_dirs:
                transcript_dir = exp_dir / "transcript"
                if not transcript_dir.exists():
                    continue
                for t in transcript_dir.glob("*.json"):
                    row = extract_row(t, correct_answers)
                    if row is not None:
                        f.write(json.dumps(row) + "\n")
                        count += 1
                        if count % 10000 == 0:
                            print(f"    {count} rows...", flush=True)

        total += count
        print(f"  → wrote {count} rows to {out_path.name}", flush=True)

    print(f"\nDone. Total rows: {total}")
    print(f"Output: {OUT_DIR}")


if __name__ == "__main__":
    main()
