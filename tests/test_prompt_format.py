"""Ensure serving prompts match training/finetune_spider.py conventions."""

from __future__ import annotations

from app.serving import SYSTEM_PROMPT, build_messages, build_prompt
from training import finetune_spider as train


def test_system_prompt_matches_training() -> None:
    assert SYSTEM_PROMPT == train.SYSTEM


def test_messages_match_training_shape() -> None:
    question = "How many singers?"
    schema = "singer(id, name)"
    msgs = build_messages(question, schema)
    assert msgs[0] == {"role": "system", "content": train.SYSTEM}
    assert msgs[1]["role"] == "user"
    assert msgs[1]["content"] == (
        f"Database schema:\n{schema}\n\nQuestion: {question}\nSQL:"
    )


def test_build_prompt_includes_roles() -> None:
    text = build_prompt("q", "s")
    assert "[system]" in text
    assert "[user]" in text
    assert "SQL:" in text
