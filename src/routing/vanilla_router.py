"""Vanilla routing strategy with full visibility and round-based turns."""

from typing import List, Dict, Any


class VanillaRouter:
    """Vanilla routing: all agents see all messages from previous round.

    This is the simplest routing strategy where:
    - All agents speak in each round (in config order)
    - Each agent sees all messages from the previous round
    - Conversations continue for a fixed number of rounds
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize vanilla router.

        Args:
            config: Routing configuration with max_rounds and strategy
        """
        self.strategy = config.get("strategy", "vanilla")
        self.max_rounds = config.get("max_rounds", 3)

        if self.max_rounds < 1:
            raise ValueError(f"max_rounds must be >= 1, got {self.max_rounds}")

    def should_continue(self, round_id: int) -> bool:
        """Check if conversation should continue to next round.

        Args:
            round_id: Current round number (0-indexed)

        Returns:
            True if should continue, False otherwise
        """
        return round_id < self.max_rounds

    def get_speaking_order(self, agent_ids: List[str], round_id: int) -> List[str]:
        """Get speaking order for a round.

        For vanilla routing, the order is always the same as defined in config.

        Args:
            agent_ids: List of agent IDs
            round_id: Current round number (unused for vanilla)

        Returns:
            List of agent IDs in speaking order
        """
        return agent_ids  # Same order every round

    def get_visible_messages(
        self,
        round_id: int,
        conversation_rounds: List[Dict[str, Any]],
        current_agent_id: str,
    ) -> List[Dict[str, Any]]:
        """Get messages visible to an agent in current round.

        For vanilla routing:
        - Round 0: No previous messages (only sees the question)
        - Round 1+: All messages from the immediately previous round
                   where current agent is in the visible_to list

        Args:
            round_id: Current round number (0-indexed)
            conversation_rounds: All completed rounds so far
            current_agent_id: ID of the agent whose turn it is

        Returns:
            List of messages visible to this agent
        """
        if round_id == 0:
            # First round - no previous messages to show
            return []

        # Get messages from previous round
        prev_round_idx = round_id - 1
        if prev_round_idx < len(conversation_rounds):
            prev_round = conversation_rounds[prev_round_idx]
            messages = prev_round.get("messages", [])

            # Filter messages where current agent is in visible_to list
            visible_messages = [
                msg for msg in messages if current_agent_id in msg.get("visible_to", [])
            ]

            return visible_messages

        return []

    def get_visibility_list(
        self,
        sender_agent_id: str,
        sender_role: str,
        all_agent_ids: List[str],
        round_id: int,
    ) -> List[str]:
        """Get list of agents who can see this message in the next round.

        For vanilla routing with participants:
        - All agents see all messages (full visibility)

        Future extensions could implement:
        - Role-based visibility (e.g., judge messages only to moderator)
        - Round-based visibility changes
        - Selective visibility based on agent relationships

        Args:
            sender_agent_id: ID of the agent sending this message
            sender_role: Role of the sender (participant, judge, moderator, etc.)
            all_agent_ids: List of all agent IDs in the conversation
            round_id: Current round number

        Returns:
            List of agent IDs who will see this message in the next round
        """
        # For vanilla routing, everyone sees everything
        return all_agent_ids

    def __repr__(self) -> str:
        """String representation."""
        return f"VanillaRouter(strategy={self.strategy}, max_rounds={self.max_rounds})"
