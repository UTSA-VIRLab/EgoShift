from __future__ import annotations

import sys
import unittest
from pathlib import Path

RELEASE = Path(__file__).resolve().parents[1]
if str(RELEASE / "tools") not in sys.path:
    sys.path.insert(0, str(RELEASE / "tools"))

from run_model_panel import credit_balance_exhausted  # noqa: E402


class ApiFailurePolicyTests(unittest.TestCase):
    def test_empty_credit_balance_is_fatal_until_billing_changes(self) -> None:
        self.assertTrue(credit_balance_exhausted("code: credit_balance_exhausted"))
        self.assertTrue(credit_balance_exhausted("You have no credits remaining."))

    def test_transient_and_daily_quotas_are_not_credit_exhaustion(self) -> None:
        self.assertFalse(credit_balance_exhausted("429 requests per day; retry in 4h"))
        self.assertFalse(credit_balance_exhausted("529 overloaded"))
        self.assertFalse(credit_balance_exhausted(None))


if __name__ == "__main__":
    unittest.main()
