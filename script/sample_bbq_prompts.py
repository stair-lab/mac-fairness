import json
import random
import os

# Set random seed for reproducibility
random.seed(42)

# Define directories relative to project root
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
data_dir = os.path.join(project_root, 'data', 'BBQ')
sampled_dir = os.path.join(data_dir, "sampled")

# Create sampled directory
os.makedirs(sampled_dir, exist_ok=True)

# Process each JSONL file
for file in os.listdir(data_dir):
    if file.endswith('.jsonl'):
        file_path = os.path.join(data_dir, file)
        with open(file_path, 'r') as f:
            lines = f.readlines()
        
        # Parse JSON lines
        data = [json.loads(line.strip()) for line in lines]
        
        # Sample 100 items
        sample_size = min(100, len(data))
        sampled = random.sample(data, sample_size)
        
        # Create output file name
        sampled_file = file.replace('.jsonl', '_sampled.jsonl')
        sampled_path = os.path.join(sampled_dir, sampled_file)
        
        # Write sampled data to new file
        with open(sampled_path, 'w') as f:
            for item in sampled:
                f.write(json.dumps(item) + '\n')
        
        print(f"Sampled {sample_size} prompts from {file} to {sampled_file}")

print("Sampling complete.")