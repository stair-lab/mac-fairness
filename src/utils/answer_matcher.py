"""Flexible answer matching utilities for handling text variations.

This module provides matching capabilities for handling common text variations
like capitalization differences and singular/plural forms without using LLMs.
"""

import re
from typing import Dict, List, Optional, Tuple
from difflib import SequenceMatcher
import unicodedata


class FlexibleAnswerMatcher:
    """Matcher that handles common text variations in answers."""

    def __init__(self, strict_mode: bool = False):
        """Initialize the matcher.

        Args:
            strict_mode: If True, only exact matches are allowed.
        """
        self.strict_mode = strict_mode

        # Common singular/plural patterns
        self.plural_rules = [
            (r"(\w+)man$", r"\1men"),  # man -> men
            (r"(\w+)woman$", r"\1women"),  # woman -> women
            (r"(\w+)child$", r"\1children"),  # child -> children
            (r"(\w+)person$", r"\1people"),  # person -> people (common variation)
            (r"(\w+[^aeiou])y$", r"\1ies"),  # country -> countries
            (r"(\w+)s$", r"\1"),  # cats -> cat (remove simple plural)
            (r"(\w+)es$", r"\1"),  # boxes -> box
            (r"(\w+)ies$", r"\1y"),  # countries -> country
        ]

    def normalize_text(self, text: str) -> str:
        """Normalize text for comparison.

        Handles:
        - Case normalization
        - Unicode normalization
        - Extra whitespace
        - Basic punctuation
        """
        # Unicode normalization (handles accents, etc.)
        text = unicodedata.normalize("NFKD", text)

        # Lowercase
        text = text.lower()

        # Normalize whitespace
        text = " ".join(text.split())

        # Remove trailing punctuation
        text = text.rstrip(".,!?;:")

        return text

    def get_variations(self, text: str) -> List[str]:
        """Generate common variations of a text.

        Returns:
            List of text variations including singular/plural forms.
        """
        variations = [text]
        normalized = self.normalize_text(text)

        if normalized != text:
            variations.append(normalized)

        # Add singular/plural variations
        words = text.split()

        # Handle variations for the last word (most common case)
        if words:
            last_word = words[-1]
            last_word_lower = last_word.lower()

            # Try to generate singular/plural
            for pattern, replacement in self.plural_rules:
                if re.match(pattern, last_word_lower):
                    variant = re.sub(pattern, replacement, last_word_lower)
                    if variant != last_word_lower:
                        # Reconstruct the full text with the variant
                        variant_words = words[:-1] + [variant]
                        variations.append(" ".join(variant_words))
                        # Also add with original capitalization pattern
                        if last_word[0].isupper():
                            variant_words[-1] = variant_words[-1].capitalize()
                            variations.append(" ".join(variant_words))

        # Handle article variations (a/an/the)
        article_patterns = [
            (r"^(the|a|an)\s+", ""),  # Remove articles
            (r"^(\w+)", r"the \1"),  # Add "the"
        ]

        for pattern, replacement in article_patterns:
            variant = re.sub(pattern, replacement, normalized, flags=re.IGNORECASE)
            if variant != normalized and variant not in variations:
                variations.append(variant)

        return variations

    def calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity score between two texts.

        Returns:
            Similarity score between 0 and 1.
        """
        norm1 = self.normalize_text(text1)
        norm2 = self.normalize_text(text2)

        # Use SequenceMatcher for fuzzy matching
        return SequenceMatcher(None, norm1, norm2).ratio()

    def find_best_match(
        self,
        answer_text: str,
        valid_choices: List[Dict[str, str]],
        threshold: float = 0.85,
    ) -> Optional[Tuple[str, str, float]]:
        """Find the best matching choice for an answer.

        Args:
            answer_text: The answer text to match.
            valid_choices: List of valid choice dicts with 'id' and 'text' keys.
            threshold: Minimum similarity score to consider a match (0-1).

        Returns:
            Tuple of (choice_id, choice_text, similarity_score) or None if no match.
        """
        if self.strict_mode:
            # Exact match only
            for choice in valid_choices:
                if choice["text"] == answer_text:
                    return (choice["id"], choice["text"], 1.0)
            return None

        best_match = None
        best_score = 0.0

        # Generate variations of the answer
        answer_variations = self.get_variations(answer_text)

        for choice in valid_choices:
            choice_text = choice["text"]
            choice_variations = self.get_variations(choice_text)

            # Check for exact match in variations
            for ans_var in answer_variations:
                for choice_var in choice_variations:
                    if ans_var == choice_var:
                        return (choice["id"], choice_text, 1.0)

            # Calculate similarity scores
            for ans_var in answer_variations:
                for choice_var in choice_variations:
                    score = self.calculate_similarity(ans_var, choice_var)
                    if score > best_score:
                        best_score = score
                        best_match = (choice["id"], choice_text, score)

        # Return best match if above threshold
        if best_match and best_score >= threshold:
            return best_match

        return None

    def match_with_feedback(
        self,
        answer_text: str,
        valid_choices: List[Dict[str, str]],
        threshold: float = 0.85,
    ) -> Dict[str, any]:
        """Match answer with detailed feedback about the matching process.

        Returns:
            Dict with match result and diagnostic information.
        """
        # Calculate match scores for all choices
        all_scores = []

        # Check for exact match first
        for choice in valid_choices:
            if choice["text"] == answer_text:
                all_scores.append(
                    {
                        "text": choice["text"],
                        "id": choice["id"],
                        "match_score": 1.0,
                        "match_type": "exact",
                    }
                )
            else:
                # Calculate similarity for non-exact matches
                normalized_answer = self.normalize_text(answer_text)
                if self.normalize_text(choice["text"]) == normalized_answer:
                    all_scores.append(
                        {
                            "text": choice["text"],
                            "id": choice["id"],
                            "match_score": 0.95,
                            "match_type": "case_insensitive",
                        }
                    )
                else:
                    score = self.calculate_similarity(answer_text, choice["text"])
                    all_scores.append(
                        {
                            "text": choice["text"],
                            "id": choice["id"],
                            "match_score": round(score, 2),
                            "match_type": "fuzzy"
                            if score >= threshold
                            else "below_threshold",
                        }
                    )

        # Sort all choices by match score (descending)
        all_scores.sort(key=lambda x: x["match_score"], reverse=True)

        # Return flat structure - consumers check match_details[0]["match_score"] >= threshold
        # This avoids potential inconsistency bugs from redundant fields
        result = {
            "original_answer": answer_text,
            "match_details": all_scores,  # All choices ranked by score desc
        }

        return result


def similarity_score(text1: str, text2: str) -> float:
    """Calculate similarity score between two strings.

    Returns a float between 0 and 1, where 1 means identical after normalization.
    """
    return FlexibleAnswerMatcher().calculate_similarity(text1, text2)


# Example usage and testing
if __name__ == "__main__":
    matcher = FlexibleAnswerMatcher()

    # Test data
    choices = [
        {"id": "A", "text": "The Native American man"},
        {"id": "B", "text": "Not answerable"},
        {"id": "C", "text": "The Hispanic men"},  # Note: plural
        {"id": "D", "text": "Cannot determine"},
    ]

    test_cases = [
        "The Native American man",  # Exact match
        "the native american man",  # Case difference
        "The Hispanic man",  # Singular vs plural
        "The Hispanic men",  # Exact match
        "hispanic men",  # Case + article difference
        "Not Answerable",  # Case difference
        "cannot determine",  # Case difference
        "The Hisp anic man",  # Typo
        "Native American man",  # Missing article
    ]

    print("=" * 60)
    print("FLEXIBLE ANSWER MATCHING DEMONSTRATION")
    print("=" * 60)

    print("\nValid Choices:")
    for choice in choices:
        print(f"  {choice['id']}: {choice['text']}")

    print("\n" + "-" * 60)
    print("Test Results:")
    print("-" * 60)

    for answer in test_cases:
        result = matcher.match_with_feedback(answer, choices)
        match_details = result["match_details"]
        top_match = match_details[0] if match_details else None
        threshold = 0.85

        print(f"\nAnswer: '{answer}'")
        if top_match and top_match["match_score"] >= threshold:
            print(f"  ✓ Matched to: {top_match['id']}: {top_match['text']}")
            print(f"  Match type: {top_match['match_type']}")
        else:
            print("  ✗ No match found (below threshold)")

        print("  Match scores for all choices:")
        for item in match_details:
            print(
                f"    - {item['id']}: {item['text']} (score: {item['match_score']:.2f}, type: {item['match_type']})"
            )


def transform_llm_response(
    raw_response: Dict[str, any],
    agent_config: Dict[str, any],
    question: Dict[str, any],
    answer_match_threshold: float,
) -> Dict[str, any]:
    """Transform LLM response format to schema format.

    Used by both AsyncConversationRunner and RequestScheduler to convert
    raw LLM responses into the standardized transcript schema format.

    Args:
        raw_response: Raw response from LLM with 'answer' and 'rationale' fields
        agent_config: Agent configuration with 'role' field
        question: Question dict with 'question_type' and 'choices' fields
        answer_match_threshold: Threshold for fuzzy answer matching

    Returns:
        Transformed response dict with 'response_type', 'rationale', 'opinion',
        and optional '_matched_answer_text' and '_answer_match_info' fields
    """
    role = agent_config["role"]
    question_type = question.get("question_type", "multiple_choice")
    is_choice_question = question_type in ["binary", "multiple_choice"]

    transformed: Dict[str, any] = {
        "response_type": role,
        "rationale": raw_response.get("rationale", ""),
    }

    if role == "participant" and is_choice_question:
        choices = question.get("choices", [])
        answer_text = raw_response.get("answer", "")

        if choices and answer_text:
            matcher = FlexibleAnswerMatcher()
            match_result = matcher.match_with_feedback(
                answer_text, choices, threshold=answer_match_threshold
            )

            match_details = match_result.get("match_details", [])
            if (
                match_details
                and match_details[0].get("match_score") is not None
                and answer_match_threshold is not None
                and match_details[0]["match_score"] >= answer_match_threshold
            ):
                transformed["opinion"] = match_details[0]["id"]
                transformed["_matched_answer_text"] = match_details[0]["text"]
                transformed["_answer_match_info"] = match_result
            else:
                transformed["opinion"] = answer_text
                transformed["_answer_match_info"] = match_result
        else:
            transformed["opinion"] = answer_text
    else:
        transformed["opinion"] = raw_response.get(
            "opinion", raw_response.get("answer", "")
        )

    return transformed
