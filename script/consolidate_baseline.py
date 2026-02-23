"""
Consolidate 1-agent baseline transcripts into per-subcategory JSONL files.

Output: experiment/consolidated_baseline/{subcategory}.jsonl
Each row: {transcript_id, question_id, model, benchmark_subcategory, experiment_name,
           status, final_answer, rationale, opinion, created_at}
"""
import json
import os
import re
import sys
from pathlib import Path

WORKSPACE = Path(os.environ.get("MAC_FAIRNESS_WORKSPACE", "."))
EXP_ROOT = Path(os.environ.get("MAC_FAIRNESS_EXPERIMENT_ROOT", WORKSPACE / "experiment"))
OUT_DIR = EXP_ROOT / "consolidated_baseline"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PAT = re.compile(r"\d{8}T\d{6}\.\d{3}Z_(.+?)_1agent_as-ai")

def extract_row(transcript_path: Path) -> dict | None:
    try:
        data = json.loads(transcript_path.read_bytes())
    except Exception:
        return None

    meta = data.get("experiment_metadata", {})
    summary = data.get("conversation_summary", {})
    experiment_name = meta.get("experiment_name", "")

    m = MODEL_PAT.search(experiment_name)
    model = m.group(1) if m else "unknown"

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

    return {
        "transcript_id": data.get("transcript_id"),
        "question_id": meta.get("question_id"),
        "model": model,
        "benchmark_subcategory": meta.get("benchmark_subcategory"),
        "experiment_name": experiment_name,
        "status": summary.get("status"),
        "final_answer": final_answer,
        "opinion": opinion,
        "rationale": rationale,
        "total_rounds": summary.get("total_rounds"),
        "created_at": data.get("created_at"),
    }


def main():
    subcats = sorted([d.name for d in EXP_ROOT.iterdir() if d.is_dir() and not d.name.startswith("consolidated")])
    print(f"Found {len(subcats)} subcategories")

    total = 0
    for subcat in subcats:
        subcat_dir = EXP_ROOT / subcat
        # Only 1-agent baseline directories
        exp_dirs = [d for d in subcat_dir.iterdir()
                    if d.is_dir() and "1agent_as-ai" in d.name]
        print(f"\n  {subcat}: {len(exp_dirs)} experiment dirs", flush=True)

        out_path = OUT_DIR / f"{subcat}.jsonl"
        count = 0
        with out_path.open("w") as f:
            for exp_dir in exp_dirs:
                transcript_dir = exp_dir / "transcript"
                if not transcript_dir.exists():
                    continue
                for t in transcript_dir.glob("*.json"):
                    row = extract_row(t)
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
