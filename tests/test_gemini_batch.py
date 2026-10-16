import json

from pathlib import Path
from unittest.mock import patch

from tools.run_gemini_batch import (
    completed_rows,
    configure_variant,
    output_text,
    request_and_meta,
    usage_dict,
)


def test_batch_response_converts_to_canonical_record():
    payload = (json.dumps({
        "key": "C:item:0",
        "response": {
            "candidates": [{"content": {"parts": [
                {"text": "private reasoning", "thought": True},
                {"text": "Visible"},
            ]}}],
            "usageMetadata": {
                "promptTokenCount": 20,
                "candidatesTokenCount": 1,
                "thoughtsTokenCount": 7,
            },
        },
    }) + "\n").encode()
    metadata = [{
        "key": "C:item:0", "task": "C", "item_id": "item",
        "permutation": 0,
        "option_order": [
            "Visible — It is visible in the current view, either fully or partially.",
            "Occluded — It is in frame but completely hidden by another object.",
        ],
        "prompt_sha256": "abc",
    }]
    row = completed_rows(payload, metadata)[0]
    assert row["raw"] == "Visible"
    assert row["parse_mode"] == "final_label"
    assert row["usage"]["output_tokens"] == 8
    assert row["execution"] == "gemini_batch_api"


def test_output_text_excludes_thought_parts():
    response = {"candidates": [{"content": {"parts": [
        {"text": "hidden", "thought": True}, {"text": "answer"}
    ]}}]}
    assert output_text(response) == "answer"


def test_usage_accepts_current_response_token_field():
    assert usage_dict({"usage_metadata": {
        "prompt_token_count": 3, "response_token_count": 2,
        "thoughts_token_count": 5,
    }}) == {
        "input_tokens": 3, "output_tokens": 7,
        "visible_output_tokens": 2, "thinking_tokens": 5,
        "stop_reason": None,
    }


def test_thinking_variant_embeds_high_level_in_batch_request():
    configure_variant("Gemini 3 Pro (thinking)", "2_reasoning", "high")
    question = {
        "id": "item", "options": ["yes", "no"], "frames": [],
        "question": "Choose.",
    }
    manifest = {"decision_rule_text": "Rule", "practice_detail": []}
    with patch("tools.run_gemini_batch.question_map",
               return_value={"item": question}), patch(
                   "tools.run_gemini_batch.load_manifest",
                   return_value=manifest):
        request, _ = request_and_meta("A", "item")
    assert request["request"]["generationConfig"] == {
        "thinkingConfig": {"thinkingLevel": "HIGH"}
    }
    configure_variant("Gemini 3 Pro", "1_frontier_api", None)
