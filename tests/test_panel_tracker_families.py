from pathlib import Path

from tools.build_panel_tracker import product_family, recency_version
import pytest

# The generated tracker doc is built in the working repository; the reviewer
# bundle ships the tools and their unit tests but not the generated markdown.
# These two checks assert properties of that generated file, so they run in
# the repository and skip where it is absent. The logic tests above always run.
requires_generated_doc = pytest.mark.skipif(
    not Path("docs/Model_Panel_Tracker.md").exists(),
    reason="docs/Model_Panel_Tracker.md not present (generated in-repo, not shipped in the bundle)",
)


def test_product_family_classifies_provider_and_open_weight_lines():
    assert product_family("GPT-5.6 Sol (thinking)") == "GPT"
    assert product_family("Claude Opus 5 (extended)") == "Claude"
    assert product_family("Gemini 3 Pro") == "Gemini"
    assert product_family("InternVL3-78B") == "InternVL"
    assert product_family("QVQ-72B-Preview") == "Qwen"
    assert product_family("SenseNova-SI-1.3-Qwen3-VL-8B") == "SenseNova"
    assert product_family("EgoMind-7B") == "EgoMind"
    assert product_family("Molmo2-8B") == "Molmo"


def test_recency_versions_do_not_confuse_parameter_count_with_generation():
    assert recency_version("GPT-5.6 Sol") > recency_version("GPT-5.1")
    assert recency_version("Gemini 3 Pro", "gemini-3.1-pro-preview") > \
        recency_version("Gemini-2.5-Pro", "gemini-2.5-pro")
    assert recency_version("Gemini 3 Flash", "gemini-3.7-flash") > \
        recency_version("Gemini 3 Pro", "gemini-3.1-pro-preview")
    assert recency_version("Qwen3-VL-8B") > recency_version("Qwen2.5-VL-72B")
    assert recency_version("QVQ-72B-Preview") == (2, 5, 0)
    assert recency_version("LLaVA-OneVision-7B") > \
        recency_version("LLaVA-1.6-Mistral-7B")
    assert recency_version("Molmo2-8B") == (2, 0, 0)


@requires_generated_doc
def test_generated_tier_tables_omit_mmsi_column():
    tracker = Path("docs/Model_Panel_Tracker.md").read_text(encoding="utf-8")
    assert "| MMSI |" not in tracker


@requires_generated_doc
def test_detailed_control_tables_have_no_empty_or_inapplicable_cells():
    tracker = Path("docs/Model_Panel_Tracker.md").read_text(encoding="utf-8")
    controls = tracker.split("### Control-arm results by model", 1)[1]
    controls = controls.split("## Task B response bias", 1)[0]
    assert "&mdash;" not in controls
    assert "n/a" not in controls
