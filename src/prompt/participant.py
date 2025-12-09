"""Participant role prompt builder implementation."""

from typing import Dict, List, Optional, Any
from .base import BasePromptBuilder


class ParticipantPromptBuilder(BasePromptBuilder):
    """Prompt builder for participant role agents."""

    def build_system_prompt(self, agent_config: Dict[str, Any]) -> str:
        """Build system prompt for participant role.

        Args:
            agent_config: Agent configuration

        Returns:
            System prompt string
        """
        role = agent_config["role"]
        persona = agent_config.get("persona")
        demographics = agent_config.get("demographics")
        if_as_human = agent_config["if_as_human"]

        # Build identity components
        identity_parts = []
        if demographics:
            identity_parts.append(demographics)
        if persona:
            identity_parts.append(persona)
        elif demographics:
            # Have demographics but no persona
            identity_parts.append("person")

        # Construct identity string
        if identity_parts:
            identity = " ".join(identity_parts)
        else:
            identity = "person"

        # Build system prompt based on if_as_human flag
        if if_as_human:
            return f"You are a {identity} acting as a {role} of the conversation."
        else:
            return f"You are an AI agent assisting a {identity} acting as a {role} of the conversation."

    def build_structured_output_schema(
        self, question: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Build structured output schema for participant responses.

        Args:
            question: Question dictionary

        Returns:
            Schema definition for participant response
        """
        question_type = question.get("question_type", "multiple_choice")
        choices = question.get("choices", [])

        # Determine valid opinion values based on question type
        if question_type in ["binary", "multiple_choice"]:
            # Extract valid choice IDs (capital letters only)
            valid_ids = [
                c["id"]
                for c in choices
                if c.get("id") and c["id"].isupper() and c["id"].isalpha()
            ]
            opinion_schema = {
                "type": "string",
                "enum": valid_ids if valid_ids else ["A", "B", "C"],  # Fallback
                "description": "Selected choice ID",
            }
        else:  # open_ended or unknown
            opinion_schema = {
                "type": "string",
                "minLength": 1,
                "description": "Free-form answer",
            }

        return {
            "type": "object",
            "required": ["response_type", "opinion", "rationale"],
            "properties": {
                "response_type": {"type": "string", "const": "participant"},
                "opinion": opinion_schema,
                "rationale": {
                    "type": "string",
                    "minLength": 1,
                    "description": "Reasoning for the opinion",
                },
                "confidence": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 1.0,
                    "description": "Optional confidence score",
                },
                "references": {
                    "type": "array",
                    "items": {"type": "string", "pattern": "^msg_\\d+_\\d{3}$"},
                    "description": "Optional references to previous messages",
                },
            },
        }

    def build_full_prompt(
        self,
        agent_config: Dict[str, Any],
        question: Dict[str, Any],
        identity_reveal_config: Dict[str, bool],
        visible_messages: Optional[List[Dict[str, Any]]] = None,
        all_agent_configs: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> str:
        """Build complete prompt for participant agent.

        Order for Round 0: System → Context → Question → Format
        Order for Round 1+: System → Context → Question → Discussion → Re-ask with Format

        Args:
            agent_config: Current agent's configuration
            question: Question dictionary
            identity_reveal_config: Identity reveal settings
            visible_messages: Previous messages visible to this agent
            all_agent_configs: All agents' configurations

        Returns:
            Complete prompt string
        """
        sections = []

        # 1. System prompt (always first)
        sections.append(self.build_system_prompt(agent_config))

        # 2. Context (if present)
        if question.get("context"):
            sections.append(f"\nContext: {question['context']}")

        # 3. Question and choices (show early so agents understand the task)
        sections.append("\n" + self._format_question_only(question))

        # 4. Previous discussion (if any - only in round 1+)
        if visible_messages and all_agent_configs:
            history = self._format_conversation_history(
                visible_messages, identity_reveal_config, all_agent_configs, question
            )
            if history:
                sections.append("\n" + history)

        # 5. Format requirements (different for round 0 vs round 1+)
        if visible_messages and all_agent_configs:
            # Round 1+: Re-ask the question with format requirements
            sections.append(
                "\n" + self._build_output_instructions_with_question(question)
            )
        else:
            # Round 0: Simple format requirements
            sections.append("\n" + self._build_output_instructions(question))

        return "".join(sections)

    def _format_question_only(self, question: Dict[str, Any]) -> str:
        """Format question and choices only (without context).

        Args:
            question: Question dictionary

        Returns:
            Formatted question string
        """
        parts = []

        # Add question text
        parts.append(f"Question: {question['question']}")

        # Add choices for multiple choice/binary questions
        question_type = question.get("question_type", "multiple_choice")
        if question_type in ["binary", "multiple_choice"]:
            choices = question.get("choices", [])
            # Only show choices with valid IDs (capital letters)
            valid_choices = [
                c
                for c in choices
                if c.get("id") and c["id"].isupper() and c["id"].isalpha()
            ]
            if valid_choices:
                parts.append("Choices:")
                for choice in valid_choices:
                    # Use "A: Text" format
                    # parts.append(f"{choice['id']}: {choice['text']}")

                    # Use "Text" format, omitting choice id
                    parts.append(f"- {choice['text']}")

        return "\n".join(parts) + "\n"

    def _format_conversation_history(
        self,
        messages: List[Dict[str, Any]],
        identity_reveal_config: Dict[str, bool],
        all_agent_configs: Dict[str, Dict[str, Any]],
        question: Dict[str, Any],
    ) -> str:
        """Format previous messages for display.

        Args:
            messages: List of previous messages
            identity_reveal_config: Identity reveal settings
            all_agent_configs: All agents' configurations
            question: Question dictionary for choice mapping

        Returns:
            Formatted conversation history
        """
        if not messages:
            return ""

        parts = [
            "Previous discussion:",
        ]

        # Build choice mapping from ID to text for natural language display
        choice_map = {}
        if question.get("choices"):
            for choice in question["choices"]:
                if choice.get("id"):
                    choice_map[choice["id"]] = choice["text"]

        for msg in messages:
            agent_id = msg.get("agent_id", "unknown")
            agent_config = all_agent_configs.get(agent_id, {"agent_id": agent_id})

            # Get identity display
            identity = self.build_agent_identity_display(
                agent_config, identity_reveal_config
            )

            # Extract response content (already validated by Zod)
            structured = msg.get("structured_response", {})
            metadata = msg.get("message_metadata", {})

            # Build display based on response type
            response_type = structured.get("response_type", "unknown")

            if response_type == "participant":
                opinion = structured.get("opinion", "?")
                # Use matched answer text from metadata if available (cleaned up version)
                answer_text = metadata.get("matched_answer_text")
                if not answer_text:
                    # Fallback: Convert letter code to full text for natural display
                    answer_text = choice_map.get(
                        opinion, opinion
                    )  # Fallback to letter if not found
                rationale = structured.get("rationale", "")
                # Natural format based on identity reveal
                if identity is None:
                    # No identity shown at all
                    parts.append(
                        f'- {agent_id}\'s the answer to the question is "{answer_text}" and the reason is that: {rationale}'
                    )
                else:
                    # Show identity in parentheses
                    parts.append(
                        f'- {agent_id} ({identity})\'s answer to the question is "{answer_text}" and the reason is that: {rationale}'
                    )
            elif response_type == "judge":
                # TODO Extend routing mechanism and debug prompt template
                verdict = structured.get("verdict", "?")
                rationale = structured.get("rationale", "")
                if identity is None:
                    parts.append(f"{agent_id}: Verdict: {verdict}. {rationale}")
                else:
                    parts.append(
                        f"{agent_id} ({identity}): Verdict: {verdict}. {rationale}"
                    )
            elif response_type == "moderator":
                # TODO Extend routing mechanism and debug prompt template
                summary = structured.get("summary", "")
                if identity is None:
                    parts.append(f"{agent_id}: {summary}")
                else:
                    parts.append(f"{agent_id} ({identity}): {summary}")
            elif response_type == "devils_advocate":
                # TODO Extend routing mechanism and debug prompt template
                challenge = structured.get("challenge", "")
                target = structured.get("target_position", "")
                if identity is None:
                    parts.append(f"{agent_id}: Challenging '{target}': {challenge}")
                else:
                    parts.append(
                        f"{agent_id} ({identity}): Challenging '{target}': {challenge}"
                    )
            else:
                # Fallback for unknown response types
                if identity is None:
                    parts.append(f"{agent_id}: [Response type: {response_type}]")
                else:
                    parts.append(
                        f"{agent_id} ({identity}): [Response type: {response_type}]"
                    )

        return "\n".join(parts) + "\n"

    def _build_output_instructions(self, question: Dict[str, Any]) -> str:
        """Build instructions for structured JSON output (Round 0).

        Args:
            question: Question dictionary

        Returns:
            Output format instructions
        """
        question_type = question.get("question_type", "multiple_choice")
        choices = question.get("choices", [])

        # Determine valid answer format using full choice text
        if question_type in ["binary", "multiple_choice"]:
            valid_choices = [
                c
                for c in choices
                if c.get("id") and c["id"].isupper() and c["id"].isalpha()
            ]
            if valid_choices:
                answer_format = (
                    "You must choose one from the following options: "
                    + " or ".join(f'"{c["text"]}"' for c in valid_choices)
                )
            else:
                answer_format = '"your selected choice"'  # Fallback
        else:
            answer_format = "your answer as a string"

        return f"""Respond with a JSON object in this exact format:
{{
  "rationale": Your reasoning behind your answer in clear and concise sentences,
  "answer": {answer_format},
}}
Output ONLY the JSON object, no other text or markdown formatting."""

    def _build_output_instructions_with_question(self, question: Dict[str, Any]) -> str:
        """Build instructions for structured JSON output with re-asking (Round 1+).

        Args:
            question: Question dictionary

        Returns:
            Output format instructions with question re-asking
        """
        question_type = question.get("question_type", "multiple_choice")
        choices = question.get("choices", [])

        # Determine valid answer format using full choice text
        if question_type in ["binary", "multiple_choice"]:
            valid_choices = [
                c
                for c in choices
                if c.get("id") and c["id"].isupper() and c["id"].isalpha()
            ]
            if valid_choices:
                answer_format = (
                    "You must choose one from the following options: "
                    + " or ".join(f'"{c["text"]}"' for c in valid_choices)
                )
            else:
                answer_format = '"your selected choice"'  # Fallback
        else:
            answer_format = "your answer as a string"

        return f"""When answering the question, respond with a JSON object in this exact format:
{{
  "rationale": Your reasoning behind your answer in clear and concise sentences,
  "answer": {answer_format},
}}
Output ONLY the JSON object, no other text or markdown formatting."""
