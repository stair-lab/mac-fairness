"""Zod schema validation via subprocess for strict type checking."""

import json
import subprocess
from pathlib import Path
from typing import Dict, Any

from src.utils.errors import (
    ZodValidationError,
    DependencyError,
    ProjectRootError,
    FileNotFoundError_,
)


class ZodValidator:
    """Strict Zod validation via subprocess - no graceful degradation."""

    def __init__(self, schema_version: str):
        """Initialize validator with schema version.

        Args:
            schema_version: Schema version to use (e.g., "2025-11-27")
        """
        self.schema_version = schema_version
        # Find project root by looking for pyproject.toml
        current = Path(__file__).resolve()
        while current != current.parent:
            if (current / "pyproject.toml").exists():
                self.project_root = current
                break
            current = current.parent
        else:
            raise ProjectRootError()

        self.validator_path = (
            self.project_root / "schema" / schema_version / "validate.ts"
        )

        if not self.validator_path.exists():
            raise FileNotFoundError_(
                file_path=str(self.validator_path), operation="load_validator"
            )

        # Test that tsx is available (try direct tsx first, then npx tsx)
        self.tsx_command = None
        for cmd in [["tsx", "--version"], ["npx", "tsx", "--version"]]:
            try:
                subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    check=True,
                    timeout=30,
                )
                # Store the working command (without --version)
                self.tsx_command = cmd[:-1]
                break
            except (
                subprocess.CalledProcessError,
                FileNotFoundError,
                subprocess.TimeoutExpired,
            ):
                continue

        if self.tsx_command is None:
            raise DependencyError(
                dependency="tsx",
                message="TypeScript executor is not available. Install with: npm install -g tsx",
            )

    def validate(self, schema_type: str, data: Any) -> Dict[str, Any]:
        """Validate data against a schema type.

        Args:
            schema_type: Type of schema to validate against
            data: Data to validate

        Returns:
            Validated data (same as input if validation passes)

        Raises:
            ValidationError: If validation fails
        """
        # Run validation using detected tsx command
        # validate.ts expects: schema_type as argument, data via stdin
        result = subprocess.run(
            self.tsx_command + [str(self.validator_path), schema_type],
            input=json.dumps(data),
            capture_output=True,
            text=True,
        )

        # Parse result
        if result.returncode == 0:
            try:
                response = json.loads(result.stdout)
                if response.get("success"):
                    # Return original data if validation passed
                    return data
                else:
                    # Extract errors from response - handle both single error and errors array
                    error_detail = response.get("error") or response.get(
                        "errors", "Unknown error"
                    )
                    raise ZodValidationError(
                        message=f"Validation failed: {error_detail}",
                        schema_type=schema_type,
                        validation_errors=[response],
                    )
            except json.JSONDecodeError:
                raise ZodValidationError(
                    message=f"Invalid validator response: {result.stdout}",
                    schema_type=schema_type,
                )
        else:
            # Check for specific error patterns
            error_msg = result.stderr or result.stdout

            if "Cannot find module" in error_msg:
                raise DependencyError(
                    dependency="validator",
                    message=f"Missing dependencies for validator. Run 'cd schema/{self.schema_version} && npm install'",
                )
            elif "SyntaxError" in error_msg:
                raise ZodValidationError(
                    message=f"Validator syntax error: {error_msg}",
                    schema_type=schema_type,
                )
            else:
                raise ZodValidationError(
                    message=f"Validation subprocess failed: {error_msg}",
                    schema_type=schema_type,
                )
