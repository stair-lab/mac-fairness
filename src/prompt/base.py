"""Base classes and interfaces for prompt generation.

This module provides the foundation for role-specific prompt generation
in a model-agnostic and benchmark-agnostic way.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any


class BasePromptBuilder(ABC):
    """Abstract base class for role-specific prompt builders.

    Each role (participant, judge, moderator, devils_advocate) should
    implement this interface to generate appropriate prompts and handle
    structured output validation.
    """

    @abstractmethod
    def build_system_prompt(self, agent_config: Dict[str, Any]) -> str:
        """Build system prompt based on agent configuration.

        Args:
            agent_config: Agent configuration with role, persona, demographics, if_as_human

        Returns:
            System prompt string defining agent's identity
        """
        pass

    @abstractmethod
    def build_structured_output_schema(
        self, question: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Build the expected structured output schema for this role.

        Args:
            question: Question dictionary to determine valid response options

        Returns:
            Schema definition for expected structured output
        """
        pass

    @abstractmethod
    def build_full_prompt(
        self,
        agent_config: Dict[str, Any],
        question: Dict[str, Any],
        identity_reveal_config: Dict[str, bool],
        visible_messages: Optional[List[Dict[str, Any]]] = None,
        all_agent_configs: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> str:
        """Build complete prompt for the agent.

        Args:
            agent_config: Current agent's configuration
            question: Question dictionary with context, choices, etc.
            identity_reveal_config: What identity info to reveal (persona, demographics, if_as_human)
            visible_messages: Previous messages visible to this agent
            all_agent_configs: All agents' configurations (for identity display)

        Returns:
            Complete prompt string ready for model input
        """
        pass

    @staticmethod
    def build_agent_identity_display(
        agent_config: Dict[str, Any], identity_reveal_config: Dict[str, bool]
    ) -> str:
        """Build agent identity display string based on reveal settings.

        This is a shared utility method used across all roles.

        Special case: If reveal_presence_mode is False, return None (no identity shown).
        This should be enforced with reveal_persona=False and reveal_demographics=False.

        Args:
            agent_config: Agent configuration
            identity_reveal_config: What to reveal (all three fields required)

        Returns:
            Identity display string (e.g., "a black doctor", "an AI agent assisting a teacher")
            or None if reveal_presence_mode is False
        """
        reveal_persona = identity_reveal_config.get("reveal_persona", False)
        reveal_demographics = identity_reveal_config.get("reveal_demographics", False)
        reveal_presence_mode = identity_reveal_config.get("reveal_presence_mode", True)

        # Special case: if reveal_presence_mode is False, don't show any identity
        # This should be paired with reveal_persona=False and reveal_demographics=False
        if not reveal_presence_mode:
            # Validate that persona and demographics are also False
            if reveal_persona or reveal_demographics:
                raise ValueError(
                    "When reveal_presence_mode is False, reveal_persona and reveal_demographics must also be False"
                )
            return None  # Signal to not show identity at all

        # Extract attributes based on reveal settings
        persona = agent_config.get("persona") if reveal_persona else None
        demographics = agent_config.get("demographics") if reveal_demographics else None
        if_as_human = agent_config.get("if_as_human", True)

        # Build identity parts
        identity_parts = []
        if demographics:
            identity_parts.append(demographics)
        if persona:
            identity_parts.append(persona)
        elif demographics:
            # Have demographics but no persona - add "person"
            identity_parts.append("person")

        # Construct display string
        if identity_parts:
            identity = " ".join(identity_parts)
            if not if_as_human:
                # Show AI assistant status (only when reveal_presence_mode is True)
                return f"an AI agent assisting a {identity}"
            else:
                # Human
                article = "an" if identity[0].lower() in "aeiou" else "a"
                return f"{article} {identity}"
        else:
            # No identity parts but showing if_as_human status
            if not if_as_human:
                return "an AI agent assisting a person"
            else:
                return "a person"
