"""Configuration management utilities."""

import yaml
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
from src.utils.errors import (
    ConfigurationError,
    MissingConfigSectionError,
    InvalidConfigFieldError,
    ProjectRootError,
    FileNotFoundError_,
)

from src.utils.recording import display_path


class ConfigManager:
    """Manages experiment configuration loading, validation, and snapshots."""

    def __init__(self, config_path: str, project_root: Optional[Path] = None):
        """Initialize the config manager.

        Args:
            config_path: Path to the configuration YAML file
            project_root: Project root directory (auto-detected if None)
        """
        self.config_path = Path(config_path)

        # Find project root if not provided
        if project_root is None:
            current = Path(__file__).resolve()
            while current != current.parent:
                if (current / "pyproject.toml").exists():
                    self.project_root = current
                    break
                current = current.parent
            else:
                raise ProjectRootError()
        else:
            self.project_root = project_root

        # Make config path absolute if not already
        if not self.config_path.is_absolute():
            self.config_path = self.project_root / self.config_path

        if not self.config_path.exists():
            raise FileNotFoundError_(
                file_path=str(self.config_path), operation="load_config"
            )

    def load_and_validate(self) -> Dict[str, Any]:
        """Load and validate the configuration file.

        Returns:
            Validated configuration dictionary

        Raises:
            ValidationError: If configuration is invalid
        """
        with open(self.config_path, "r") as f:
            config = yaml.safe_load(f)

        # Validate required top-level sections for new structure
        required_sections = [
            "experiment_metadata",
            "conversation_config",
            "retry_config",
            "identity_reveal_config",
            "model_config",
            "agent_definitions",
        ]

        for section in required_sections:
            if section not in config:
                raise MissingConfigSectionError(section)

        # Validate experiment_metadata
        exp_meta = config["experiment_metadata"]
        required_meta_fields = [
            "experiment_name",
            "benchmark_subcategory",
            "schema_version",
            "questions_file",
        ]
        for field in required_meta_fields:
            if field not in exp_meta:
                raise MissingConfigSectionError(f"experiment_metadata.{field}")

        # Validate conversation_config
        conv_config = config["conversation_config"]
        conv_field_types = {
            "routing_strategy": str,
            "max_rounds": int,
        }
        for field, expected_type in conv_field_types.items():
            if field not in conv_config:
                raise MissingConfigSectionError(f"conversation_config.{field}")
            if not isinstance(conv_config[field], expected_type):
                raise InvalidConfigFieldError(
                    field=f"conversation_config.{field}",
                    expected_type=expected_type.__name__,
                    actual_type=type(conv_config[field]).__name__,
                )

        # Validate identity_reveal_config has all required boolean fields
        reveal = config["identity_reveal_config"]
        for field in ["reveal_persona", "reveal_demographics", "reveal_presence_mode"]:
            if field not in reveal:
                raise MissingConfigSectionError(f"identity_reveal_config.{field}")
            if not isinstance(reveal[field], bool):
                raise InvalidConfigFieldError(
                    field=f"identity_reveal_config.{field}",
                    expected_type="bool",
                    actual_type=type(reveal[field]).__name__,
                )

        # Special validation: if reveal_presence_mode is false, others must be false too
        if not reveal["reveal_presence_mode"]:
            if reveal["reveal_persona"] or reveal["reveal_demographics"]:
                raise ConfigurationError(
                    "When reveal_presence_mode is false, reveal_persona and reveal_demographics "
                    "must also be false. This creates completely anonymous agents."
                )
            print(
                "Anonymous mode: reveal_presence_mode=false (no identity information shown)"
            )

        # Validate experiment naming convention for identity reveal
        exp_name = exp_meta["experiment_name"]
        identity_keywords = ["as-human", "as-ai", "as-hybrid", "as-anonymous"]
        if not any(keyword in exp_name.lower() for keyword in identity_keywords):
            print(
                f"Warning: Experiment name '{exp_name}' should include one of: "
                f"{', '.join(identity_keywords)} to indicate identity reveal mode"
            )

        # Validate agents have required fields
        agents = config["agent_definitions"]
        for i, agent in enumerate(agents):
            required_agent_fields = [
                "agent_id",
                "role",
                "if_as_human",
                "model",
                "temperature",
                "max_tokens",
            ]
            for field in required_agent_fields:
                if field not in agent:
                    raise MissingConfigSectionError(f"agent_definitions[{i}].{field}")

            # Validate if_as_human is boolean
            if not isinstance(agent["if_as_human"], bool):
                raise InvalidConfigFieldError(
                    field=f"agent_definitions[{i}].if_as_human",
                    expected_type="bool",
                    actual_type=type(agent["if_as_human"]).__name__,
                )

        print(f"✓ Configuration validated: {exp_meta['experiment_name']}")
        return config

    def save_snapshot(
        self, config: Dict[str, Any], submission_timestamp: datetime
    ) -> str:
        """Save a snapshot of the configuration.

        Args:
            config: Configuration dictionary to save
            submission_timestamp: Timestamp for the snapshot

        Returns:
            Path to the saved snapshot file
        """
        timestamp = submission_timestamp.strftime("%Y%m%dT%H%M%SZ")

        exp_meta = config["experiment_metadata"]
        benchmark = exp_meta["benchmark_subcategory"]
        experiment = exp_meta["experiment_name"]

        snapshot_path = (
            self.project_root
            / "bookkeeping"
            / "config_snapshot"
            / benchmark
            / f"{experiment}_{timestamp}.yaml"
        )

        # Create directory if it doesn't exist
        snapshot_path.parent.mkdir(parents=True, exist_ok=True)

        with open(snapshot_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)

        relative_path = display_path(snapshot_path, self.project_root)
        print(f"✓ Config snapshot saved: {relative_path}")
        return relative_path
