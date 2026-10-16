.PHONY: test test-unittest smoke support-check check

# pytest is the runner: the suite mixes unittest.TestCase classes with
# pytest-style module-level tests, and `unittest discover` silently collects
# zero tests from the latter rather than reporting them.
test:
	PYTHONPATH=src:. python -m pytest tests -q

# unittest-only subset, for an environment without pytest. Collects fewer
# tests; `make test` is the authoritative run.
test-unittest:
	PYTHONPATH=src:. python -m unittest discover -s tests -v

smoke:
	PYTHONPATH=src:. python -m egoshift smoke --root sample

support-check:
	PYTHONPATH=src:. python tools/compare_support_implementations.py --config configs/taskb_geometry_sensitivity.json --enumerator tools/independent_support_enumerator.py >/dev/null

check: test smoke support-check
