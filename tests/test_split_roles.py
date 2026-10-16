from __future__ import annotations

import unittest

from egoshift.split_roles import (
    assign_component_role,
    canonical_exposure_manifest_sha256,
    require_complete_split_roles,
)


class SplitRoleTests(unittest.TestCase):
    def test_assignment_is_deterministic(self) -> None:
        first = assign_component_role("component", seed="seed", outcome_exposed=False)
        second = assign_component_role("component", seed="seed", outcome_exposed=False)
        self.assertEqual(first, second)

    def test_exposed_component_is_never_hidden(self) -> None:
        for index in range(1000):
            role = assign_component_role(
                f"component-{index}", seed="seed", outcome_exposed=True
            )
            self.assertIn(role, {"tuning_dev", "public_dev"})

    def test_split_without_complete_exposure_binding_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "complete reviewed exposure ledger"):
            require_complete_split_roles({"pass": True, "exposure_files": []})

    def test_hash_bound_complete_exposure_binding_is_accepted(self) -> None:
        records = [{"path": "history.jsonl", "sha256": "a" * 64}]
        binding = {
            "status": "complete_reviewed",
            "complete": True,
            "audit_sha256": "b" * 64,
            "manifest_sha256": canonical_exposure_manifest_sha256(records),
            "reviewed_manifest_count": 1,
        }
        observed = require_complete_split_roles(
            {
                "pass": True,
                "exposure_files": records,
                "exposure_ledger": binding,
                "exposed_component_assigned_hidden_violations": [],
            }
        )
        self.assertEqual(observed, binding)


if __name__ == "__main__":
    unittest.main()
