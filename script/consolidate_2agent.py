"""
Consolidate 2-agent multi-round transcripts into human-readable JSONL files.

Output: experiment/consolidated_2agent/{subcategory}.jsonl

Key improvements over raw transcripts:
- Identity fields (persona, demographics, if_as_human, reveal_condition) from config snapshot
- correct_answer from benchmark data
- Derived columns: position_shifted, initial_agreement, is_correct per agent
- Readable column names: identity_agent (spkr_000) vs vanilla_agent (spkr_001)
- Human-readable summary text per row
"""
import json
import os
import re
from functools import lru_cache
from pathlib import Path

import yaml

WORKSPACE = Path(os.environ.get("MAC_FAIRNESS_WORKSPACE", "."))
EXP_ROOT = Path(os.environ.get("MAC_FAIRNESS_EXPERIMENT_ROOT", WORKSPACE / "experiment"))
OUT_DIR = EXP_ROOT / "consolidated_2agent"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PAT = re.compile(r"\d{8}T\d{6}\.\d{3}Z_(.+?)_2agent_id-vanilla")


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
    """Cache config snapshot YAML reads (hundreds of unique paths, not millions)."""
    resolved = snap_path.replace("$MAC_FAIRNESS_WORKSPACE", str(WORKSPACE))
    try:
        return yaml.safe_load(Path(resolved).read_text()) or {}
    except Exception:
        return {}


def extract_identity(snap: dict) -> dict:
    """Extract identity and reveal fields from a config snapshot."""
    agents = snap.get("agent_definitions", [])
    identity = {"persona": None, "demographics": None, "if_as_human": None}
    for ag in agents:
        if ag.get("agent_id") == "spkr_000":
            identity["persona"] = ag.get("persona")
            identity["demographics"] = ag.get("demographics")
            identity["if_as_human"] = ag.get("if_as_human")
            break
    reveal = snap.get("identity_reveal_config", {})
    # Condition is "revealed" if all three are true, "anonymous" if all false
    all_revealed = all([
        reveal.get("reveal_persona"),
        reveal.get("reveal_demographics"),
        reveal.get("reveal_presence_mode"),
    ])
    all_hidden = not any([
        reveal.get("reveal_persona"),
        reveal.get("reveal_demographics"),
        reveal.get("reveal_presence_mode"),
    ])
    identity["reveal_condition"] = "revealed" if all_revealed else ("anonymous" if all_hidden else "partial")
    return identity


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

    # Load identity from config snapshot (cached)
    snap_path = meta.get("config_snapshot_path", "")
    snap = load_config_snapshot(snap_path)
    identity = extract_identity(snap)

    # Extract per-round opinions (spkr_000 = identity agent, spkr_001 = vanilla)
    rounds = {}
    for rnd in data.get("conversation_rounds", []):
        rid = rnd["round_id"]
        for msg in rnd.get("messages", []):
            agent = msg["agent_id"]
            sr = msg.get("structured_response", {})
            rounds.setdefault(rid, {})[agent] = sr.get("opinion")

    ia_r = [rounds.get(i, {}).get("spkr_000") for i in range(3)]  # identity agent per round
    va_r = [rounds.get(i, {}).get("spkr_001") for i in range(3)]  # vanilla agent per round

    final_answers = summary.get("final_answers", {})
    ia_final = final_answers.get("spkr_000")
    va_final = final_answers.get("spkr_001")
    correct = correct_answers.get(question_id)
    status = summary.get("status")
    total_rounds = summary.get("total_rounds", 0)

    # Derived fields
    ia_shifted = ia_r[0] != ia_final if (ia_r[0] and ia_final) else None
    va_shifted = va_r[0] != va_final if (va_r[0] and va_final) else None
    initial_agreement = ia_r[0] == va_r[0] if (ia_r[0] and va_r[0]) else None
    final_agreement = summary.get("consensus_reached")
    ia_correct_r0 = (ia_r[0] == correct) if (ia_r[0] and correct) else None
    ia_correct_final = (ia_final == correct) if (ia_final and correct) else None
    va_correct_r0 = (va_r[0] == correct) if (va_r[0] and correct) else None
    va_correct_final = (va_final == correct) if (va_final and correct) else None

    # Human-readable summary
    persona = identity["persona"] or "no-persona"
    demo = identity["demographics"] or "no-demo"
    framing = "as-human" if identity["if_as_human"] else "as-AI"
    reveal = identity["reveal_condition"] or "?"
    round_trace = " → ".join(
        f"R{i}:[{ia_r[i] or '?'},{va_r[i] or '?'}]"
        for i in range(total_rounds or 0)
    )
    summary_text = (
        f"[{persona} | {demo} | {framing} | {reveal}] "
        f"{round_trace} "
        f"→ final:[{ia_final or '?'},{va_final or '?'}] "
        f"correct:{correct or '?'} "
        f"ia_shifted:{ia_shifted} va_shifted:{va_shifted}"
    )

    return {
        # Identifiers
        "transcript_id": data.get("transcript_id"),
        "question_id": question_id,
        "model": model,
        "benchmark_subcategory": meta.get("benchmark_subcategory"),
        "experiment_name": experiment_name,
        # Identity condition
        "persona": identity["persona"],
        "demographics": identity["demographics"],
        "if_as_human": identity["if_as_human"],
        "reveal_condition": identity["reveal_condition"],
        # Conversation outcome
        "status": status,
        "total_rounds": total_rounds,
        "correct_answer": correct,
        "final_agreement": final_agreement,
        # Identity agent (spkr_000) per round
        "identity_agent_r0": ia_r[0],
        "identity_agent_r1": ia_r[1],
        "identity_agent_r2": ia_r[2],
        "identity_agent_final": ia_final,
        "identity_agent_shifted": ia_shifted,
        "identity_agent_correct_r0": ia_correct_r0,
        "identity_agent_correct_final": ia_correct_final,
        # Vanilla agent (spkr_001) per round
        "vanilla_agent_r0": va_r[0],
        "vanilla_agent_r1": va_r[1],
        "vanilla_agent_r2": va_r[2],
        "vanilla_agent_final": va_final,
        "vanilla_agent_shifted": va_shifted,
        "vanilla_agent_correct_r0": va_correct_r0,
        "vanilla_agent_correct_final": va_correct_final,
        # Initial state
        "initial_agreement": initial_agreement,
        # Human-readable
        "summary": summary_text,
        "created_at": data.get("created_at"),
    }


def main():
    print("Loading correct answers from benchmark data...", flush=True)
    correct_answers = load_correct_answers(WORKSPACE)
    print(f"  Loaded {len(correct_answers)} question→answer mappings", flush=True)

    subcats = sorted([
        d.name for d in EXP_ROOT.iterdir()
        if d.is_dir() and not d.name.startswith("consolidated")
    ])
    print(f"Found {len(subcats)} subcategories\n")

    total = 0
    for subcat in subcats:
        subcat_dir = EXP_ROOT / subcat
        exp_dirs = [d for d in subcat_dir.iterdir()
                    if d.is_dir() and "2agent_id-vanilla" in d.name]
        if not exp_dirs:
            continue
        print(f"  {subcat}: {len(exp_dirs)} experiment dirs", flush=True)

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
        print(f"  → wrote {count} rows to {out_path.name}\n", flush=True)

    print(f"Done. Total rows: {total}")
    print(f"Output: {OUT_DIR}")


if __name__ == "__main__":
    main()
