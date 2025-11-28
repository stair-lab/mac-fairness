"""Performance metrics and statistics utilities."""

from typing import Dict, List, Any, Optional

from src.utils.recording import aggregate_validation_errors


class MetricsCollector:
    """Collects and aggregates performance metrics for conversations."""

    def calculate_conversation_metrics(
        self, conversation_rounds: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Calculate metrics for a single conversation.

        Args:
            conversation_rounds: List of conversation round data

        Returns:
            Dictionary of calculated metrics
        """
        total_messages = sum(len(r["messages"]) for r in conversation_rounds)
        total_rounds = len(conversation_rounds)

        # Extract token counts
        total_tokens_generated = 0
        total_prompt_tokens = 0
        total_time_ms = 0

        for round_data in conversation_rounds:
            for msg in round_data["messages"]:
                metadata = msg.get("message_metadata", {})
                perf = metadata.get("performance", {})

                total_tokens_generated += perf.get("tokens_generated", 0)
                total_prompt_tokens += perf.get("prompt_tokens", 0)
                total_time_ms += perf.get("generation_time_ms", 0)

        return {
            "total_messages": total_messages,
            "total_rounds": total_rounds,
            "total_tokens_generated": total_tokens_generated,
            "total_prompt_tokens": total_prompt_tokens,
            "total_time_ms": total_time_ms,
            "average_response_time_ms": total_time_ms / total_messages
            if total_messages > 0
            else 0,
        }

    def extract_final_answers(
        self, conversation_rounds: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Extract final answers from the last round of conversation.

        Args:
            conversation_rounds: List of conversation round data

        Returns:
            Dictionary mapping agent_id to their final answer
        """
        final_answers = {}

        if conversation_rounds and conversation_rounds[-1]["messages"]:
            for msg in conversation_rounds[-1]["messages"]:
                agent_id = msg["agent_id"]
                response = msg.get("structured_response", {})

                # Extract answer based on response type
                if response.get("response_type") == "participant":
                    final_answers[agent_id] = response.get("opinion")
                elif response.get("response_type") == "judge":
                    final_answers[agent_id] = response.get("verdict")

        return final_answers

    def check_consensus(self, final_answers: Dict[str, Any]) -> Optional[bool]:
        """Check if consensus was reached among agents.

        Args:
            final_answers: Dictionary of agent final answers

        Returns:
            True if consensus reached, False if not, None if no answers
        """
        if not final_answers:
            return None

        unique_answers = set(final_answers.values())
        return len(unique_answers) == 1

    def calculate_retry_statistics(
        self, conversation_rounds: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Calculate retry statistics from conversation rounds.

        Args:
            conversation_rounds: List of conversation round data

        Returns:
            Dictionary of retry statistics
        """
        total_retry_attempts = 0
        messages_requiring_retries = 0
        max_retries_per_message = 0

        for round_data in conversation_rounds:
            for msg in round_data["messages"]:
                retry_count = msg.get("message_metadata", {}).get("retry_count", 0)
                total_retry_attempts += retry_count
                if retry_count > 0:
                    messages_requiring_retries += 1
                    max_retries_per_message = max(max_retries_per_message, retry_count)

        return {
            "total_retry_attempts": total_retry_attempts,
            "messages_requiring_retries": messages_requiring_retries,
            "max_retries_per_message": max_retries_per_message,
        }

    def aggregate_validation_errors(
        self, all_validation_errors: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Aggregate validation errors using error codes.

        Args:
            all_validation_errors: List of all validation errors

        Returns:
            List of aggregated error summaries with counts
        """
        # Delegate to shared function
        return aggregate_validation_errors(all_validation_errors)

    def calculate_experiment_summary(
        self, transcripts: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Calculate summary statistics for an entire experiment.

        Args:
            transcripts: List of conversation transcripts

        Returns:
            Dictionary of experiment-level statistics
        """
        total_conversations = len(transcripts)
        successful_conversations = sum(
            1
            for t in transcripts
            if t.get("conversation_summary", {}).get("status") == "succeeded"
        )
        failed_conversations = total_conversations - successful_conversations

        # Aggregate token usage
        total_tokens_all = sum(
            t.get("conversation_summary", {})
            .get("performance_metrics", {})
            .get("total_tokens", 0)
            for t in transcripts
        )

        # Consensus statistics
        consensus_count = sum(
            1
            for t in transcripts
            if t.get("conversation_summary", {}).get("consensus_reached") is True
        )

        # Error type aggregation
        all_errors = []
        for t in transcripts:
            errors = (
                t.get("conversation_summary", {})
                .get("retry_statistics", {})
                .get("validation_errors_summary", [])
            )
            all_errors.extend(errors)

        aggregated_errors = self.aggregate_validation_errors(all_errors)

        return {
            "total_conversations": total_conversations,
            "successful_conversations": successful_conversations,
            "failed_conversations": failed_conversations,
            "success_rate": successful_conversations / total_conversations
            if total_conversations > 0
            else 0,
            "consensus_rate": consensus_count / successful_conversations
            if successful_conversations > 0
            else 0,
            "total_tokens_used": total_tokens_all,
            "average_tokens_per_conversation": total_tokens_all / total_conversations
            if total_conversations > 0
            else 0,
            "error_summary": aggregated_errors,
        }
