"""Conversation manager for orchestrating multi-agent discussions."""

import json
import fcntl
import os
import uuid
import yaml
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

from src.agent.mock_agent import MockAgent


class ZodValidator:
    """Helper class for mandatory Zod validation via subprocess."""

    def __init__(self):
        self.schema_dir = Path("schema/2025-11-27")
        self.validate_script = self.schema_dir / "dist" / "validate.js"

        # Check validation setup on initialization
        if not self.validate_script.exists():
            raise FileNotFoundError(
                f"✗ Zod validation required but not available.\n"
                f"  Please run:\n"
                f"    cd {self.schema_dir}\n"
                f"    npm install\n"
                f"    npm run build"
            )

    def validate(
        self, schema_type: str, data: Any
    ) -> tuple[bool, Optional[Dict], Optional[List]]:
        """Validate data against a Zod schema (mandatory).

        Returns:
            Tuple of (success, validated_data, errors)

        Raises:
            RuntimeError: If validation script fails to run
            ValueError: If validation fails (when used in strict mode)
        """
        import subprocess
        import json

        try:
            result = subprocess.run(
                ["node", str(self.validate_script), schema_type],
                input=json.dumps(data),
                capture_output=True,
                text=True,
                timeout=5,
            )
            output = json.loads(result.stdout)

            if output.get("success"):
                return True, output.get("data"), None
            else:
                return False, None, output.get("errors", [])
        except subprocess.TimeoutExpired:
            raise RuntimeError(f"Zod validation timed out for {schema_type}")
        except json.JSONDecodeError as e:
            raise RuntimeError(
                f"Invalid JSON from validator: {e}\nOutput: {result.stdout}"
            )
        except FileNotFoundError:
            raise RuntimeError("Node.js is not installed or not in PATH")


class ConversationManager:
    """Manages multi-agent conversations and bookkeeping."""

    def __init__(self, config_path: str):
        """Initialize manager with configuration.

        Args:
            config_path: Path to YAML configuration file
        """
        self.config_path = config_path
        self.config = self.load_config(config_path)
        self.ensure_directories()
        self.agents = {}
        self.snapshot_path = None
        self.validator = ZodValidator()

    def load_config(self, config_path: str) -> Dict:
        """Load and validate configuration from YAML file.

        Args:
            config_path: Path to configuration file

        Returns:
            Configuration dictionary
        """
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        # Basic validation of required fields
        exp = config.get("experiment", {})
        required_fields = [
            "experiment_name",
            "benchmark_name",
            "questions_file",
            "agent_config_axes",
            "max_rounds",
            "routing_strategy",
            "agents",
        ]

        missing = [f for f in required_fields if f not in exp]
        if missing:
            print(f"✗ Config validation failed: Missing fields: {', '.join(missing)}")
            raise ValueError(f"Missing required experiment fields: {missing}")

        print(f"✓ Config loaded for: {exp['experiment_name']}")

        return config

    def ensure_directories(self):
        """Create necessary directories for bookkeeping and transcripts."""
        # Bookkeeping directories
        Path("bookkeeping").mkdir(exist_ok=True)
        Path("bookkeeping/config_snapshot").mkdir(exist_ok=True)

        benchmark = self.config["experiment"]["benchmark_name"]
        experiment = self.config["experiment"]["experiment_name"]

        # Config snapshot directory
        Path(f"bookkeeping/config_snapshot/{benchmark}").mkdir(
            exist_ok=True, parents=True
        )

        # Experiment directories (transcript and job_summary)
        Path(f"experiment/{benchmark}/{experiment}/transcript").mkdir(
            exist_ok=True, parents=True
        )
        Path(f"experiment/{benchmark}/{experiment}/job_summary").mkdir(
            exist_ok=True, parents=True
        )

    def save_config_snapshot(self) -> str:
        """Save timestamped config snapshot.

        Returns:
            Path to saved snapshot
        """
        timestamp = datetime.now().strftime("%Y%m%dT%H%M%SZ")
        benchmark = self.config["experiment"]["benchmark_name"]
        experiment = self.config["experiment"]["experiment_name"]

        snapshot_path = (
            f"bookkeeping/config_snapshot/{benchmark}/" f"{experiment}_{timestamp}.yaml"
        )

        with open(snapshot_path, "w") as f:
            yaml.dump(self.config, f, default_flow_style=False)

        self.snapshot_path = snapshot_path
        print(f"Config snapshot saved: {snapshot_path}")
        return snapshot_path

    def append_to_index(self, entry: Dict):
        """Append entry to index.jsonl with file locking.

        Args:
            entry: Dictionary containing transcript metadata
        """
        # Use benchmark-specific index file
        benchmark_name = self.config["experiment"]["benchmark_name"]
        if benchmark_name == "mocktest":
            index_path = "bookkeeping/mocktest_index.jsonl"
        else:
            index_path = "bookkeeping/index.jsonl"

        # Create file if it doesn't exist
        if not os.path.exists(index_path):
            open(index_path, "a").close()

        with open(index_path, "a") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                json.dump(entry, f, separators=(",", ":"))
                f.write("\n")
                f.flush()
                os.fsync(f.fileno())
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)

    def initialize_agents(self):
        """Initialize agents based on configuration."""
        for agent_config in self.config["experiment"]["agents"]:
            agent_id = agent_config["agent_id"]

            # For now, only support mock agents
            if self.config["experiment"]["models"]["dummy-model"]["family"] == "dummy":
                self.agents[agent_id] = MockAgent(agent_config)
            else:
                raise NotImplementedError("Only mock agents supported currently")

    def construct_prompt(
        self, agent_config: Dict, question: Dict, history: List = None
    ) -> str:
        """Construct prompt for an agent.

        Args:
            agent_config: Agent configuration
            question: Question dictionary
            history: List of previous messages

        Returns:
            Formatted prompt string
        """
        # Build system prompt based on agent attributes
        if agent_config.get("as_human", True):
            identity = f"You are a person acting as a {agent_config['role']}"
        else:
            identity = f"You are an AI agent acting as a {agent_config['role']}"

        # Format question
        question_text = question["question"]
        choices_text = "\n".join(
            f"{choice['id']}: {choice['text']}" for choice in question["choices"]
        )

        # Build full prompt
        prompt_parts = [
            identity,
            "\nQuestion:",
            question_text,
            "\nChoices:",
            choices_text,
        ]

        # Add history if this is not round 0
        if history:
            prompt_parts.append("\n\nPrevious discussion:")
            for msg in history:
                prompt_parts.append(f"{msg['agent_id']}: {msg['response']}")

        prompt_parts.append("\n\nYour response:")

        return "\n".join(prompt_parts)

    def run_conversation(self, question: Dict) -> Dict:
        """Run a multi-agent conversation for a single question.

        Args:
            question: Question dictionary from JSONL

        Returns:
            Complete conversation transcript
        """
        transcript_id = str(uuid.uuid4())
        conversation_rounds = []

        # Run conversation rounds
        max_rounds = self.config["experiment"]["max_rounds"]

        for round_num in range(max_rounds):
            round_messages = []

            # Get message history for this round
            history = []
            if round_num > 0:
                # For vanilla routing, include all previous rounds
                for prev_round in conversation_rounds:
                    history.extend(prev_round["messages"])

            # Each agent responds in order
            for agent_config in self.config["experiment"]["agents"]:
                agent_id = agent_config["agent_id"]
                agent = self.agents[agent_id]

                # Construct prompt
                prompt = self.construct_prompt(agent_config, question, history)

                # Generate response
                result = agent.generate(prompt)

                # Create message
                message = {
                    "agent_id": agent_id,
                    "role": agent_config["role"],
                    "response": result["response"],
                    "structured_output": result.get("structured_output"),
                }

                round_messages.append(message)
                history.append(message)  # Add to history for next agent

            # Store round
            conversation_rounds.append({"round": round_num, "messages": round_messages})

        # Create full transcript
        transcript = {
            "transcript_id": transcript_id,
            "protocol_version": "2025-11-27",
            "experiment_metadata": {
                "experiment_name": self.config["experiment"]["experiment_name"],
                "benchmark_name": self.config["experiment"]["benchmark_name"],
                "config_snapshot_path": self.snapshot_path,
                "submission_timestamp": datetime.now().isoformat() + "Z",
                "execution_timestamp": datetime.now().isoformat() + "Z",
            },
            "question": question,
            "routing_config": {
                "strategy": self.config["experiment"]["routing_strategy"],
                "max_rounds": max_rounds,
            },
            "agents": self.config["experiment"]["agents"],
            "conversation_rounds": conversation_rounds,
            "created_at": datetime.now().isoformat() + "Z",
        }

        # Validate transcript with Zod before saving (mandatory)
        success, validated_data, errors = self.validator.validate(
            "conversation", transcript
        )
        if not success:
            print("✗ Transcript validation failed:")
            if errors:
                for error in errors[:5]:  # Show first 5 errors
                    path = ".".join(str(p) for p in error.get("path", []))
                    print(f"  - {path}: {error.get('message', 'Unknown error')}")
            raise ValueError(f"Transcript validation failed with {len(errors)} errors")

        transcript = validated_data or transcript

        # Save transcript
        self.save_transcript(transcript)

        # Update index
        self.update_index(transcript, question)

        return transcript

    def save_transcript(self, transcript: Dict):
        """Save conversation transcript to JSON file.

        Args:
            transcript: Complete conversation transcript
        """
        benchmark = self.config["experiment"]["benchmark_name"]
        experiment = self.config["experiment"]["experiment_name"]
        transcript_id = transcript["transcript_id"]

        transcript_path = (
            f"experiment/{benchmark}/{experiment}/transcript/" f"{transcript_id}.json"
        )

        with open(transcript_path, "w") as f:
            json.dump(transcript, f, indent=2)

    def save_job_summary(
        self,
        start_time: datetime,
        end_time: datetime,
        questions_processed: int,
        question_range: tuple = None,
    ):
        """Save job execution summary with unique filename to prevent overwriting.

        Args:
            start_time: When the experiment started
            end_time: When the experiment ended
            questions_processed: Number of questions processed
            question_range: Optional range of questions processed
        """
        benchmark = self.config["experiment"]["benchmark_name"]
        experiment = self.config["experiment"]["experiment_name"]

        # Create unique filename using timestamp and job ID
        job_id = os.environ.get("SLURM_JOB_ID", "local")
        array_task_id = os.environ.get("SLURM_ARRAY_TASK_ID", None)
        timestamp = datetime.now().strftime("%Y%m%dT%H%M%SZ")

        # Construct filename with unique identifiers
        if array_task_id:
            filename = f"{timestamp}_{job_id}_{array_task_id}.json"
        else:
            # For local runs, timestamp alone ensures uniqueness
            filename = f"{timestamp}_{job_id}.json"

        summary_path = f"experiment/{benchmark}/{experiment}/job_summary/{filename}"

        # Create job summary data
        summary = {
            "job_id": job_id,
            "array_task_id": array_task_id,
            "experiment_name": experiment,
            "benchmark_name": benchmark,
            "start_time": start_time.isoformat() + "Z",
            "end_time": end_time.isoformat() + "Z",
            "duration_seconds": (end_time - start_time).total_seconds(),
            "questions_processed": questions_processed,
            "question_range": {
                "start": question_range[0] if question_range else 0,
                "end": question_range[1] if question_range else questions_processed,
            },
            "config_snapshot": self.snapshot_path,
            "hostname": os.environ.get("HOSTNAME", "unknown"),
            "working_directory": os.getcwd(),
            "environment": {
                "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                "slurm_array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID"),
                "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
                "slurm_cpus_per_task": os.environ.get("SLURM_CPUS_PER_TASK"),
                "slurm_mem_per_node": os.environ.get("SLURM_MEM_PER_NODE"),
                "slurm_gpus": os.environ.get("SLURM_GPUS"),
            },
            "created_at": timestamp,
        }

        # Save summary
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        print(f"Job summary saved: {summary_path}")

    def update_index(self, transcript: Dict, question: Dict):
        """Update index with transcript metadata.

        Args:
            transcript: Complete conversation transcript
            question: Original question
        """
        # Get job_id (same as used in job_summary)
        job_id = os.environ.get("SLURM_JOB_ID", "local")

        index_entry = {
            "transcript_id": transcript["transcript_id"],
            "experiment_name": self.config["experiment"]["experiment_name"],
            "benchmark_name": self.config["experiment"]["benchmark_name"],
            "question_id": question["id"],
            "job_id": job_id,
            "agent_config_axes": self.config["experiment"]["agent_config_axes"],
            "submission_timestamp": transcript["experiment_metadata"][
                "submission_timestamp"
            ],
            "execution_timestamp": transcript["experiment_metadata"][
                "execution_timestamp"
            ],
            "transcript_path": f"experiment/{self.config['experiment']['benchmark_name']}/{self.config['experiment']['experiment_name']}/transcript/{transcript['transcript_id']}.json",
            "config_snapshot_path": self.snapshot_path,
            "protocol_version": "2025-11-27",
            "routing_strategy": self.config["experiment"]["routing_strategy"],
            "n_agents": len(self.config["experiment"]["agents"]),
            "shared_model_backbone": self.config["experiment"].get(
                "shared_model_backbone"
            ),
            "agents": self.config["experiment"]["agents"],
        }

        # Append index entry directly (validation is optional)
        self.append_to_index(index_entry)

    def run_experiment(self, question_range: tuple = None):
        """Run complete experiment on all questions.

        Args:
            question_range: Optional (start, end) indices for processing subset
        """
        # Track experiment start time
        start_time = datetime.now()

        # Save config snapshot first
        self.save_config_snapshot()

        # Initialize agents
        self.initialize_agents()

        # Load and validate questions
        questions = []
        with open(self.config["experiment"]["questions_file"], "r") as f:
            for line_num, line in enumerate(f, 1):
                if line.strip():
                    try:
                        q_data = json.loads(line)
                        # Mandatory Zod validation for questions
                        success, validated_data, errors = self.validator.validate(
                            "question", q_data
                        )
                        if not success:
                            print(f"✗ Question validation failed at line {line_num}:")
                            if errors:
                                for error in errors[:3]:  # Show first 3 errors
                                    path = ".".join(
                                        str(p) for p in error.get("path", [])
                                    )
                                    print(f"  - {path}: {error.get('message', '')}")
                            raise ValueError(
                                f"Question at line {line_num} failed validation"
                            )

                        questions.append(validated_data or q_data)
                    except json.JSONDecodeError as e:
                        print(f"✗ Invalid JSON at line {line_num}: {e}")
                        raise

        # Apply range if specified
        if question_range:
            start, end = question_range
            questions = questions[start:end]

        print(f"Processing {len(questions)} questions...")

        # Process each question
        for i, question in enumerate(questions, 1):
            self.run_conversation(question)
            print(f"[{i}/{len(questions)}] Processed question {question['id']}")

        # Save job summary
        end_time = datetime.now()
        self.save_job_summary(start_time, end_time, len(questions), question_range)

        print(f"\nExperiment complete! Processed {len(questions)} questions.")
        print(
            f"Transcripts saved to: experiment/{self.config['experiment']['benchmark_name']}/{self.config['experiment']['experiment_name']}/transcript/"
        )

        # Show which index file was updated
        benchmark_name = self.config["experiment"]["benchmark_name"]
        if benchmark_name == "mocktest":
            print("Index updated: bookkeeping/mocktest_index.jsonl")
        else:
            print("Index updated: bookkeeping/index.jsonl")
