#!/bin/bash
# Submit experiment to Slurm with config snapshot at queuing time
#
# Usage:
#   # Single job (process all questions in one task):
#   ./script/cluster/submit_slurm.sh config/bbq_race/experiment_scratch.yaml
#
#   # Array job (divide questions among tasks):
#   ./script/cluster/submit_slurm.sh config/bbq_race/experiment_scratch.yaml --array-tasks 20 --total-questions 6879
#
#   # Array job with automatic question counting:
#   ./script/cluster/submit_slurm.sh config/bbq_race/experiment_scratch.yaml --array-tasks 20

if [ $# -lt 1 ]; then
    echo "Usage: $0 <config_file> [--array-tasks N] [--total-questions M]"
    echo "  --array-tasks N: Number of parallel tasks to divide work"
    echo "  --total-questions M: Total number of questions (auto-detected if not specified)"
    exit 1
fi

CONFIG_FILE="$1"
ARRAY_TASKS=""
TOTAL_QUESTIONS=""

# Parse arguments
shift
while [ $# -gt 0 ]; do
    case "$1" in
        --array-tasks)
            ARRAY_TASKS="$2"
            shift 2
            ;;
        --total-questions)
            TOTAL_QUESTIONS="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Check config exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file not found: $CONFIG_FILE"
    exit 1
fi

# Extract experiment details using Python one-liner
EXPERIMENT_INFO=$(python3 -c "
import yaml, sys
with open('$CONFIG_FILE') as f:
    config = yaml.safe_load(f)
    exp = config['experiment']
    print(f\"{exp['benchmark_name']} {exp['experiment_name']} {exp['questions_file']}\")
")

BENCHMARK=$(echo "$EXPERIMENT_INFO" | cut -d' ' -f1)
EXPERIMENT=$(echo "$EXPERIMENT_INFO" | cut -d' ' -f2)
QUESTIONS_FILE=$(echo "$EXPERIMENT_INFO" | cut -d' ' -f3)

# If array tasks specified but no total questions, count them
if [ -n "$ARRAY_TASKS" ] && [ -z "$TOTAL_QUESTIONS" ]; then
    if [ -f "$QUESTIONS_FILE" ]; then
        # Count only non-empty lines (matching Python's loading behavior)
        TOTAL_QUESTIONS=$(grep -c . "$QUESTIONS_FILE" || true)
        echo "Auto-detected $TOTAL_QUESTIONS non-empty lines in $QUESTIONS_FILE"

        # Verify it's a valid JSONL by checking first line
        FIRST_LINE=$(head -n1 "$QUESTIONS_FILE")
        if ! echo "$FIRST_LINE" | python3 -m json.tool > /dev/null 2>&1; then
            echo "Warning: First line doesn't appear to be valid JSON"
            echo "  $FIRST_LINE"
        fi
    else
        echo "Error: Cannot find questions file: $QUESTIONS_FILE"
        echo "Please specify --total-questions manually"
        exit 1
    fi
fi

# Create timestamp
TIMESTAMP=$(date -u +"%Y%m%dT%H%M%SZ")

# Create snapshot directory
SNAPSHOT_DIR="bookkeeping/config_snapshot/${BENCHMARK}"
mkdir -p "$SNAPSHOT_DIR"

# Create snapshot at queuing time
SNAPSHOT_PATH="${SNAPSHOT_DIR}/${EXPERIMENT}_${TIMESTAMP}.yaml"
cp "$CONFIG_FILE" "$SNAPSHOT_PATH"

echo "Config snapshot saved at queuing time: $SNAPSHOT_PATH"

# Create slurm logs directory
mkdir -p "slurm_logs/${BENCHMARK}"

# Generate Slurm script
SLURM_SCRIPT="/tmp/slurm_${EXPERIMENT}_$$.sh"

cat > "$SLURM_SCRIPT" << EOF
#!/bin/bash
#SBATCH --job-name=${EXPERIMENT}
#SBATCH --output=slurm_logs/${BENCHMARK}/${EXPERIMENT}_%A_%a.out
#SBATCH --error=slurm_logs/${BENCHMARK}/${EXPERIMENT}_%A_%a.err
#SBATCH --time=4:00:00
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
EOF

# Add array specification if provided
if [ -n "$ARRAY_TASKS" ]; then
    # Create array job with task IDs from 0 to N-1
    echo "#SBATCH --array=0-$((ARRAY_TASKS - 1))" >> "$SLURM_SCRIPT"
fi

cat >> "$SLURM_SCRIPT" << EOF

# Activate virtual environment
source .venv/bin/activate

# Set experiments root if needed
export MAC_FAIRNESS_EXPERIMENT_ROOT=\${MAC_FAIRNESS_EXPERIMENT_ROOT:-./experiment}

# Run experiment using the snapshot (not the scratch config)
EOF

# Add appropriate command based on array or single job
if [ -n "$ARRAY_TASKS" ]; then
    cat >> "$SLURM_SCRIPT" << EOF

# Calculate question range for this array task
# Divide $TOTAL_QUESTIONS questions among $ARRAY_TASKS tasks
TOTAL_Q=$TOTAL_QUESTIONS
NUM_TASKS=$ARRAY_TASKS
QUESTIONS_PER_TASK=\$((TOTAL_Q / NUM_TASKS))
REMAINDER=\$((TOTAL_Q % NUM_TASKS))

# Calculate start and end for this task
# First REMAINDER tasks get an extra question
if [ \$SLURM_ARRAY_TASK_ID -lt \$REMAINDER ]; then
    START=\$((SLURM_ARRAY_TASK_ID * (QUESTIONS_PER_TASK + 1)))
    END=\$((START + QUESTIONS_PER_TASK + 1))
else
    START=\$((REMAINDER * (QUESTIONS_PER_TASK + 1) + (SLURM_ARRAY_TASK_ID - REMAINDER) * QUESTIONS_PER_TASK))
    END=\$((START + QUESTIONS_PER_TASK))
fi

echo "Task \$SLURM_ARRAY_TASK_ID: Processing questions \$START to \$((END - 1))"
python script/run_experiment.py ${SNAPSHOT_PATH} --range \$START-\$END
EOF
else
    echo "python script/run_experiment.py ${SNAPSHOT_PATH}" >> "$SLURM_SCRIPT"
fi

# Submit job
JOB_ID=$(sbatch "$SLURM_SCRIPT" | awk '{print $4}')

# Clean up temp script
rm "$SLURM_SCRIPT"

echo ""
echo "================== Job Submitted Successfully =================="
echo "Snapshot: $SNAPSHOT_PATH"
echo "Job ID: $JOB_ID"

if [ -n "$ARRAY_TASKS" ]; then
    echo "Array tasks: $ARRAY_TASKS"
    echo "Total questions: $TOTAL_QUESTIONS"
    QUESTIONS_PER_TASK=$((TOTAL_QUESTIONS / ARRAY_TASKS))
    REMAINDER=$((TOTAL_QUESTIONS % ARRAY_TASKS))
    echo "Questions per task: ~$QUESTIONS_PER_TASK"
    if [ $REMAINDER -gt 0 ]; then
        echo "  (First $REMAINDER tasks process $((QUESTIONS_PER_TASK + 1)) questions)"
    fi
fi

echo ""
echo "You can now safely edit $CONFIG_FILE"
echo "The queued job will use the snapshot, not the scratch file."
echo "================================================================"