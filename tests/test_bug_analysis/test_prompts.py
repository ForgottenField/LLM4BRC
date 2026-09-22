"""Tests for prompt templates."""

from llm_client.bug_analysis.prompts import (
    FEW_SHOT_EXAMPLES,
    build_messages,
    build_system_prompt,
)


class TestPrompts:
    def test_system_prompt_non_empty(self):
        prompt = build_system_prompt()
        assert len(prompt) > 100
        assert "Output Schema" in prompt
        assert "bug_type" in prompt

    def test_few_shot_examples_count(self):
        """Should have 4 examples (SARIF, plain text, JSON, POC viability)."""
        assert len(FEW_SHOT_EXAMPLES) >= 8  # 4 pairs of user+assistant

    def test_few_shot_roles_alternate(self):
        """Roles should alternate: user, assistant, user, assistant, ..."""
        for i, msg in enumerate(FEW_SHOT_EXAMPLES):
            expected = "user" if i % 2 == 0 else "assistant"
            assert msg["role"] == expected, f"Message {i} expected role {expected}"

    def test_build_messages_structure(self):
        messages = build_messages("test report content")
        # system + few-shot + user
        assert messages[0]["role"] == "system"
        assert messages[-1]["role"] == "user"
        assert messages[-1]["content"] == "test report content"
        # Verify every message has both keys
        for msg in messages:
            assert "role" in msg
            assert "content" in msg

    def test_build_messages_places_report_last(self):
        messages = build_messages("BUGBUG")
        assert messages[-1]["content"] == "BUGBUG"
