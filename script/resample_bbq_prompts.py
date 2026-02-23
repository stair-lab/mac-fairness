import argparse
import json
import os
import random

# Set random seed for reproducibility
# used seed 50 for resampling nationality, physical_appearance, and sexual_orientation BBQ files; used seed 42 otherwise
random.seed(50)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Sample BBQ prompts with context_condition='disambig' from a specified JSONL file."
    )
    parser.add_argument(
        "--bbq-file",
        required=True,
        help="Input BBQ JSONL filename (e.g., bbq_age.jsonl) located in data/BBQ.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=100,
        help="Number of items to sample (default: 100).",
    )
    parser.add_argument(
        "--context-condition",
        choices=["ambig", "disambig"],
        default="disambig",
        help="Context condition to sample (default: disambig).",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Define directories relative to project root
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    data_dir = os.path.join(project_root, "data", "BBQ")
    sampled_dir = os.path.join(data_dir, "new_sampled")

    # Create sampled directory
    os.makedirs(sampled_dir, exist_ok=True)

    file_path = os.path.join(data_dir, args.bbq_file)
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"BBQ file not found: {file_path}")

    with open(file_path, "r") as f:
        lines = f.readlines()

    # Parse and filter JSON lines
    data = [json.loads(line.strip()) for line in lines]
    filtered_data = [
        item
        for item in data
        if item.get("source_metadata", {}).get("context_condition")
        == args.context_condition
    ]

    # Sample items
    sample_size = min(args.sample_size, len(filtered_data))
    sampled = random.sample(filtered_data, sample_size) if sample_size > 0 else []

    # Create output file name
    base_name = os.path.splitext(args.bbq_file)[0]
    sampled_file = f"{base_name}_sampled.jsonl"
    sampled_path = os.path.join(sampled_dir, sampled_file)

    # Write sampled data to new file
    with open(sampled_path, "w") as f:
        for item in sampled:
            f.write(json.dumps(item) + "\n")

    print(
        f"Sampled {sample_size} {args.context_condition} prompts from {args.bbq_file} "
        f"to {sampled_file}"
    )


if __name__ == "__main__":
    main()