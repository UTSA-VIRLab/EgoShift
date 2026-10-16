from pathlib import Path

from tools.build_human_tracker import class_pct
import pytest

# The generated tracker doc is built in the working repository; the reviewer
# bundle ships the tools and their unit tests but not the generated markdown.
# These two checks assert properties of that generated file, so they run in
# the repository and skip where it is absent. The logic tests above always run.
requires_generated_doc = pytest.mark.skipif(
    not Path("docs/Human_Annotation_Results.md").exists(),
    reason="docs/Human_Annotation_Results.md not present (generated in-repo, not shipped in the bundle)",
)


def test_class_accuracy_is_rendered_as_percentage():
    assert class_pct({"accuracy": 1.0, "correct": 60, "of": 60}) == "100.0%"
    assert class_pct({"accuracy": 0.9167, "correct": 55, "of": 60}) == "91.7%"


def test_class_accuracy_falls_back_to_counts_and_handles_missing():
    assert class_pct({"correct": 30, "of": 40}) == "75.0%"
    assert class_pct(None) == "&mdash;"


@requires_generated_doc
def test_generated_sessions_table_omits_time_span():
    tracker = Path("docs/Human_Annotation_Results.md").read_text(encoding="utf-8")
    sessions = tracker.split("## Sessions", 1)[1].split("### Task A", 1)[0]
    assert "Time span" not in sessions


@requires_generated_doc
def test_generated_tracker_omits_task_order_summary():
    tracker = Path("docs/Human_Annotation_Results.md").read_text(encoding="utf-8")
    assert "Actual task order from embedded timestamps" not in tracker
