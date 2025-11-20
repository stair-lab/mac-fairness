"""Mock agent for testing without GPU/model requirements."""


class MockAgent:
    """Mock agent that returns deterministic responses for testing."""

    def __init__(self, agent_config):
        """Initialize mock agent with configuration.

        Args:
            agent_config: Dictionary containing agent configuration
                - agent_id: Unique identifier (e.g., "speaker_001")
                - role: Agent role (e.g., "participant", "judge")
                - as_human: Whether to present as human (bool)
                - model: Model name (should be "shared" for mock)
        """
        self.config = agent_config
        self.agent_id = agent_config.get("agent_id", "unknown")
        self.role = agent_config.get("role", "participant")
        self.as_human = agent_config.get("as_human", True)

    def generate(self, prompt, temperature=0.7, max_tokens=512):
        """Generate mock response.

        Args:
            prompt: Input prompt for the agent
            temperature: Sampling temperature (ignored for mock)
            max_tokens: Maximum tokens to generate (ignored for mock)

        Returns:
            Dictionary containing:
                - response: Text response
                - structured_output: Structured answer if applicable
        """
        # For testing, always choose "C" (Cannot determine) for safety
        return {
            "response": f"As {self.agent_id} ({self.role}), considering the question: {prompt[:50]}... I believe the answer is C because we cannot make determinations without sufficient evidence.",
            "structured_output": {
                "answer": "C",
                "reasoning": f"Mock response from {self.agent_id}: Cannot determine from given information",
            },
        }
