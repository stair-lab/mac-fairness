"""Grid configuration expansion for running parameter sweeps.

This module provides functionality to expand a single "grid config" file
into multiple individual configurations for parameter sweep experiments.

Usage:
    # In a grid config YAML file, add a _grid section:
    _grid:
      derive:
        questions_file: "data/BBQ/{benchmark_subcategory}.jsonl"
      sweep:
        benchmark_subcategory: [bbq_race, bbq_gender]
        prompt_template_config.for_participant.choice_display_format: [letter_dot, bullet]

    # This generates 2 x 2 = 4 configurations with all combinations
"""

import copy
import itertools
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

from src.utils.logging import info_print


def _get_nested_value(config: Dict[str, Any], key_path: str) -> Any:
    """Get a value from a nested dictionary using dot notation.

    Args:
        config: Configuration dictionary
        key_path: Dot-separated path (e.g., "prompt_template_config.for_participant.choice_display_format")

    Returns:
        Value at the path

    Raises:
        KeyError: If path doesn't exist
    """
    keys = key_path.split(".")
    value = config
    for key in keys:
        value = value[key]
    return value


def _set_nested_value(config: Dict[str, Any], key_path: str, value: Any) -> None:
    """Set a value in a nested dictionary using dot notation.

    Args:
        config: Configuration dictionary (modified in place)
        key_path: Dot-separated path
        value: Value to set
    """
    keys = key_path.split(".")
    obj = config
    for key in keys[:-1]:
        if key not in obj:
            obj[key] = {}
        obj = obj[key]
    obj[keys[-1]] = value


def _apply_derivation(template: str, config: Dict[str, Any]) -> str:
    """Apply a derivation template using config values.

    Args:
        template: Template string with {field.path} placeholders (dot notation required)
        config: Configuration dictionary to get values from

    Returns:
        Resolved string with placeholders replaced

    Example:
        >>> _apply_derivation("data/BBQ/{experiment_metadata.benchmark_subcategory}.jsonl",
        ...                   {"experiment_metadata": {"benchmark_subcategory": "bbq_race"}})
        "data/BBQ/bbq_race.jsonl"
    """
    # Find all {field} placeholders
    pattern = r"\{([^}]+)\}"
    matches = re.findall(pattern, template)

    result = template
    for field in matches:
        try:
            value = _get_nested_value(config, field)
        except (KeyError, TypeError):
            raise ValueError(
                f"Cannot resolve derivation placeholder '{{{field}}}': "
                f"field not found in config. Use full dot-notation path."
            )

        result = result.replace(f"{{{field}}}", str(value))

    return result


def _generate_experiment_name_suffix(sweep_values: Dict[str, Any]) -> str:
    """Generate a suffix for experiment name based on sweep values.

    Args:
        sweep_values: Dictionary of parameter path -> value

    Returns:
        Suffix string for experiment name
    """
    parts = []
    for key, value in sweep_values.items():
        # Use the last part of the key path as abbreviation
        short_key = key.split(".")[-1]
        # Abbreviate common keys
        abbrevs = {
            "benchmark_subcategory": "bench",
            "choice_display_format": "fmt",
            "json_field_order": "ord",
        }
        short_key = abbrevs.get(short_key, short_key[:4])
        parts.append(f"{short_key}={value}")

    return "_".join(parts)


class GridConfigExpander:
    """Expands grid configuration into multiple individual configurations."""

    GRID_KEY = "_grid"
    SWEEP_KEY = "sweep"
    DERIVE_KEY = "derive"

    def __init__(self, config_path: str):
        """Initialize with a configuration file path.

        Args:
            config_path: Path to the grid configuration YAML file
        """
        self.config_path = Path(config_path)

        with open(self.config_path, "r") as f:
            self.raw_config = yaml.safe_load(f)

    def is_grid_config(self) -> bool:
        """Check if this configuration is a grid config.

        Returns:
            True if the config has a _grid section with sweep parameters
        """
        if self.GRID_KEY not in self.raw_config:
            return False

        grid_spec = self.raw_config[self.GRID_KEY]
        return self.SWEEP_KEY in grid_spec and bool(grid_spec[self.SWEEP_KEY])

    def get_sweep_parameters(self) -> Dict[str, List[Any]]:
        """Get the sweep parameters and their values.

        Returns:
            Dictionary mapping parameter paths to lists of values
        """
        if not self.is_grid_config():
            return {}

        return self.raw_config[self.GRID_KEY].get(self.SWEEP_KEY, {})

    def get_derivation_rules(self) -> Dict[str, str]:
        """Get the derivation rules.

        Returns:
            Dictionary mapping target fields to template strings
        """
        if self.GRID_KEY not in self.raw_config:
            return {}

        return self.raw_config[self.GRID_KEY].get(self.DERIVE_KEY, {})

    def get_num_combinations(self) -> int:
        """Get the total number of configurations that will be generated.

        Returns:
            Number of configuration combinations
        """
        sweep_params = self.get_sweep_parameters()
        if not sweep_params:
            return 1

        num = 1
        for values in sweep_params.values():
            num *= len(values)
        return num

    def expand(self) -> List[Tuple[Dict[str, Any], Dict[str, Any]]]:
        """Expand the grid config into individual configurations.

        Returns:
            List of (config_dict, sweep_specs_dict) tuples where sweep_specs_dict
            maps full config paths to their values for this combination.

        Raises:
            ValueError: If grid configuration is invalid
        """
        if not self.is_grid_config():
            # Not a grid config, return as-is
            base_config = copy.deepcopy(self.raw_config)
            return [(base_config, {})]

        sweep_params = self.get_sweep_parameters()
        derive_rules = self.get_derivation_rules()

        # Create base config without _grid section
        base_config = {k: v for k, v in self.raw_config.items() if k != self.GRID_KEY}

        # Generate all combinations
        param_names = list(sweep_params.keys())
        param_values = [sweep_params[name] for name in param_names]

        configs = []
        for combination in itertools.product(*param_values):
            # Create a copy of base config
            config = copy.deepcopy(base_config)

            # Apply sweep values - sweep_values maps full path to value
            sweep_values = {}
            for name, value in zip(param_names, combination):
                _set_nested_value(config, name, value)
                sweep_values[name] = value

            # Apply derivation rules
            for target_field, template in derive_rules.items():
                derived_value = _apply_derivation(template, config)
                _set_nested_value(config, target_field, derived_value)

            configs.append((config, sweep_values))

        return configs

    def expand_with_sweep_specs(self) -> List[Tuple[Dict[str, Any], Dict[str, Any]]]:
        """Expand grid config into individual configurations with sweep specs.

        Returns:
            List of (config_dict, grid_sweep_specs) tuples where grid_sweep_specs
            is a dict mapping full config paths to their active values.
        """
        return self.expand()

    def print_summary(self) -> None:
        """Print a summary of the grid configuration."""
        if not self.is_grid_config():
            info_print("Not a grid configuration (no _grid.sweep section)")
            return

        sweep_params = self.get_sweep_parameters()
        derive_rules = self.get_derivation_rules()

        info_print("=" * 60, prefix=False)
        info_print("Grid Configuration Summary", prefix=False)
        info_print("=" * 60, prefix=False)

        info_print("Sweep Parameters:", prefix=False)
        for param, values in sweep_params.items():
            info_print(f"  {param}:", prefix=False)
            for v in values:
                info_print(f"    - {v}", prefix=False)

        if derive_rules:
            info_print("Derivation Rules:", prefix=False)
            for target, template in derive_rules.items():
                info_print(f"  {target}: {template}", prefix=False)

        info_print(f"Total combinations: {self.get_num_combinations()}")

    def print_expanded_configs(self, verbose: bool = False) -> None:
        """Print all expanded configurations.

        Args:
            verbose: If True, print full config; otherwise just names
        """
        expanded = self.expand_with_sweep_specs()

        info_print("=" * 60, prefix=False)
        info_print(f"Expanded Configurations ({len(expanded)} total)", prefix=False)
        info_print("=" * 60, prefix=False)

        for i, (config, sweep_specs) in enumerate(expanded, 1):
            exp_name = config["experiment_metadata"]["experiment_name"]
            info_print(f"[{i}] {exp_name}", prefix=False)
            info_print(f"    experiment_metadata.questions_file: {config['experiment_metadata']['questions_file']}", prefix=False)

            # Show sweep parameter values from sweep_specs
            for param_path, value in sweep_specs.items():
                info_print(f"    {param_path}: {value}", prefix=False)

            if verbose:
                info_print("    Full config:", prefix=False)
                yaml_str = yaml.dump(config, default_flow_style=False)
                for line in yaml_str.split("\n"):
                    info_print(f"      {line}", prefix=False)


def load_grid_or_regular_config(
    config_path: str, is_grid: bool = False
) -> List[Tuple[Dict[str, Any], Dict[str, Any]]]:
    """Load a configuration file, expanding if it's a grid config.

    Args:
        config_path: Path to configuration file
        is_grid: If True, treat as grid config; if False, load as regular config

    Returns:
        List of (config_dict, grid_sweep_specs) tuples
    """
    expander = GridConfigExpander(config_path)

    if is_grid:
        if not expander.is_grid_config():
            raise ValueError(
                f"--grid flag specified but config has no _grid.sweep section: {config_path}"
            )
        return expander.expand_with_sweep_specs()
    else:
        if expander.is_grid_config():
            raise ValueError(
                f"Config has _grid section but --grid flag not specified. "
                f"Use --grid to expand, or remove _grid section: {config_path}"
            )
        # Return as single config with empty sweep specs
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        return [(config, {})]
