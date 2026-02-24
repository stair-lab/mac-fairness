"""
Push consolidated experiment results to HuggingFace dataset repo.

Usage:
  python script/push_to_hf.py                    # push both baseline + 2-agent
  python script/push_to_hf.py --baseline          # push only baseline
  python script/push_to_hf.py --2agent            # push only 2-agent
  python script/push_to_hf.py --consolidate       # re-consolidate from raw transcripts before pushing
  python script/push_to_hf.py --consolidate-only  # only consolidate, don't push

Requires:
  - HF token at /lfs/skampere1/0/sttruong/.cache/huggingface/token (or HF_TOKEN env var)
  - huggingface_hub installed
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

WORKSPACE = Path(os.environ.get("MAC_FAIRNESS_WORKSPACE", Path(__file__).resolve().parent.parent))
EXP_ROOT = Path(os.environ.get("MAC_FAIRNESS_EXPERIMENT_ROOT", WORKSPACE / "experiment"))
REPO_ID = "aims-foundation/mac-fairness"
TOKEN_PATH = Path("/lfs/skampere1/0/sttruong/.cache/huggingface/token")
PYTHON = sys.executable


def get_token() -> str:
    token = os.environ.get("HF_TOKEN")
    if token:
        return token.strip()
    if TOKEN_PATH.exists():
        return TOKEN_PATH.read_text().strip()
    raise RuntimeError(f"No HF token found. Set HF_TOKEN env var or place token at {TOKEN_PATH}")


def consolidate(which: str):
    """Run consolidation script(s)."""
    env = {**os.environ, "MAC_FAIRNESS_WORKSPACE": str(WORKSPACE)}
    if which in ("baseline", "both"):
        script = WORKSPACE / "script" / "consolidate_baseline.py"
        if script.exists():
            print(f"\n{'='*60}\nConsolidating baseline transcripts...\n{'='*60}")
            subprocess.run([PYTHON, "-u", str(script)], env=env, check=True)
    if which in ("2agent", "both"):
        script = WORKSPACE / "script" / "consolidate_2agent.py"
        if script.exists():
            print(f"\n{'='*60}\nConsolidating 2-agent transcripts...\n{'='*60}")
            subprocess.run([PYTHON, "-u", str(script)], env=env, check=True)


def push(which: str, token: str):
    """Upload consolidated JSONL files to HuggingFace."""
    from huggingface_hub import HfApi
    api = HfApi(token=token)

    if which in ("baseline", "both"):
        folder = EXP_ROOT / "consolidated_baseline"
        if folder.exists() and any(folder.glob("*.jsonl")):
            count = sum(1 for _ in folder.glob("*.jsonl"))
            print(f"\nUploading {count} baseline JSONL files from {folder}...")
            result = api.upload_folder(
                folder_path=str(folder),
                repo_id=REPO_ID,
                repo_type="dataset",
                path_in_repo="baseline/bbq-sampled",
                commit_message=f"Update baseline BBQ results ({count} subcategories)",
            )
            print(f"Done: {result}")
        else:
            print(f"No baseline data at {folder}, skipping.")

    if which in ("2agent", "both"):
        folder = EXP_ROOT / "consolidated_2agent"
        if folder.exists() and any(folder.glob("*.jsonl")):
            count = sum(1 for _ in folder.glob("*.jsonl"))
            print(f"\nUploading {count} 2-agent JSONL files from {folder}...")
            result = api.upload_folder(
                folder_path=str(folder),
                repo_id=REPO_ID,
                repo_type="dataset",
                path_in_repo="2agent-vanilla/bbq-sampled",
                commit_message=f"Update 2-agent BBQ results ({count} subcategories)",
            )
            print(f"Done: {result}")
        else:
            print(f"No 2-agent data at {folder}, skipping.")


def main():
    parser = argparse.ArgumentParser(description="Push experiment results to HuggingFace")
    parser.add_argument("--baseline", action="store_true", help="Push only baseline results")
    parser.add_argument("--2agent", dest="two_agent", action="store_true", help="Push only 2-agent results")
    parser.add_argument("--consolidate", action="store_true", help="Re-consolidate from raw transcripts before pushing")
    parser.add_argument("--consolidate-only", action="store_true", help="Only consolidate, don't push to HF")
    args = parser.parse_args()

    which = "both"
    if args.baseline and not args.two_agent:
        which = "baseline"
    elif args.two_agent and not args.baseline:
        which = "2agent"

    if args.consolidate or args.consolidate_only:
        consolidate(which)
        if args.consolidate_only:
            print("\nConsolidation complete. Skipping push (--consolidate-only).")
            return

    token = get_token()
    push(which, token)


if __name__ == "__main__":
    main()
