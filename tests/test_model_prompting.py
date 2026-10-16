from __future__ import annotations

import sys
import unittest
from pathlib import Path

RELEASE = Path(__file__).resolve().parents[1]
if str(RELEASE / "src") not in sys.path:
    sys.path.insert(0, str(RELEASE / "src"))

from egoshift.model_prompting import build_prompt, parse_answer  # noqa: E402

ITEM = {
    "id": "x1",
    "question": "Where is the sofa?",
    "options": ["in front", "to the right", "behind", "to the left"],
    "frames": ["a.png", "b.png"],
}
MANIFEST = {
    "decision_rule_text": (
        "You are the camera. Zones are measured from the direction you are now "
        "facing: in front is less than 45 degrees off-centre; behind is more "
        "than 135 degrees either way. Every question is clear of a boundary."
    ),
    "practice_detail": [{"id": "p1", "answer": "behind", "reason": "It swung past 135."}],
}


class PromptTests(unittest.TestCase):
    def test_main_arm_carries_rule_question_and_options(self) -> None:
        p = build_prompt(ITEM, MANIFEST)
        self.assertIn("You are the camera", p["text"])
        self.assertIn("Where is the sofa?", p["text"])
        for option in ITEM["options"]:
            self.assertIn(option, p["text"])
        self.assertEqual(p["images"], ["a.png", "b.png"])

    def test_no_image_arm_keeps_text_identical_but_drops_images(self) -> None:
        a = build_prompt(ITEM, MANIFEST)
        b = build_prompt(ITEM, MANIFEST, arm="no_image")
        self.assertEqual(a["text"], b["text"])
        self.assertEqual(b["images"], [])

    def test_single_view_arm_keeps_only_the_first_frame(self) -> None:
        self.assertEqual(build_prompt(ITEM, MANIFEST, arm="single_view")["images"], ["a.png"])

    def test_pp1_removes_only_the_zone_sentence(self) -> None:
        p = build_prompt(ITEM, MANIFEST, arm="pp1_no_rule")
        self.assertIn("You are the camera", p["text"])
        self.assertNotIn("less than 45 degrees", p["text"])

    def test_pp2_adds_the_annotators_worked_examples(self) -> None:
        p = build_prompt(ITEM, MANIFEST, arm="pp2_few_shot")
        self.assertIn("Worked examples", p["text"])
        self.assertIn("It swung past 135.", p["text"])

    def test_prompt_hash_tracks_option_order(self) -> None:
        a = build_prompt(ITEM, MANIFEST)
        b = build_prompt(ITEM, MANIFEST, option_order=list(reversed(ITEM["options"])))
        self.assertNotEqual(a["prompt_sha256"], b["prompt_sha256"])

    def test_parse_handles_verbatim_case_letter_and_index(self) -> None:
        opts = ITEM["options"]
        self.assertEqual(parse_answer("behind", opts)[0], "behind")
        self.assertEqual(parse_answer("  Behind ", opts)[0], "behind")
        self.assertEqual(parse_answer("The answer is behind.", opts)[0], "behind")
        self.assertEqual(parse_answer("C", opts)[0], "behind")
        self.assertEqual(parse_answer("3", opts)[0], "behind")

    def test_parse_refuses_to_guess_when_ambiguous(self) -> None:
        self.assertIsNone(parse_answer("maybe left or right", ITEM["options"])[0])
        self.assertIsNone(parse_answer("", ITEM["options"])[0])

    def test_parse_uses_only_terminal_answer_after_reasoning(self) -> None:
        raw = (
            "I considered in front and behind, so an anywhere-in-text match is ambiguous.\n"
            "Final answer: behind"
        )
        self.assertEqual(parse_answer(raw, ITEM["options"]), ("behind", "final_marker"))

    def test_parse_accepts_qvq_terminal_latex_box_only(self) -> None:
        raw = (
            "I considered behind and to the left during the reasoning.\n\n"
            "**Final Answer**\n\n"
            r"\[ \boxed{\text{in front}} \]"
        )
        self.assertEqual(
            parse_answer(raw, ITEM["options"]), ("in front", "final_boxed"))
        # A boxed option in intermediate reasoning must not override an
        # ambiguous terminal line.
        unsafe = (r"Maybe \boxed{\text{behind}}, but perhaps in front."
                  + "\nStill uncertain.")
        self.assertIsNone(parse_answer(unsafe, ITEM["options"])[0])


if __name__ == "__main__":
    unittest.main()
