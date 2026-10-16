"""Match the object-type distribution across Task-C visibility classes.

Occlusion is not type-neutral in a furnished room. Small soft objects get
covered because furniture can cover them; large fixed objects stay visible
because nothing can. Left alone that correlation leaks straight into the
question text, which names the object: on a balanced pool, a rule reading only
the noun scored 64.4% against 33.3% chance.

This module removes the leak by construction rather than measuring it later.
For every object type it keeps the same number of items in every class, so the
type marginal is identical across classes and the noun carries exactly zero
information about the answer. Types that cannot appear in all three classes are
dropped entirely -- a type present in only one class is the purest form of the
shortcut.

The cost is yield: matching is bounded by the scarcest class per type, and most
of the surplus is discarded. That is the intended trade.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Callable, Iterable, Mapping, Sequence


def object_type_of(target_oid: str) -> str:
    """The type key used for matching: the objectId prefix."""

    return str(target_oid).split("|", 1)[0]


def type_oracle_accuracy(
    labelled: Sequence[tuple[str, str]],
) -> tuple[float, float]:
    """Accuracy of predicting the class from the object type alone.

    Returns ``(oracle, chance)`` where chance is the majority-class rate. On a
    type-matched pool the oracle must equal chance: every type is equally
    likely to carry any class, so the best a type-only rule can do is guess.
    """

    if not labelled:
        return 0.0, 0.0
    by_type: dict[str, Counter] = defaultdict(Counter)
    overall: Counter = Counter()
    for object_type, answer in labelled:
        by_type[object_type][answer] += 1
        overall[answer] += 1
    correct = sum(max(counts.values()) for counts in by_type.values())
    chance = max(overall.values()) / len(labelled)
    return correct / len(labelled), chance


def match_object_type_distribution(
    candidates: Iterable[Mapping[str, Any]],
    *,
    classes: Sequence[str],
    class_of: Callable[[Mapping[str, Any]], str],
    type_of: Callable[[Mapping[str, Any]], str],
    order_key: Callable[[Mapping[str, Any]], Any],
    per_class_cap: int | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Keep an equal number of every object type in every class.

    ``per_class_cap`` optionally trims the matched pool to a target size while
    preserving the match: types are reduced evenly, largest first, so no type
    becomes class-informative on the way down.
    """

    grouped: dict[str, dict[str, list[Mapping[str, Any]]]] = defaultdict(
        lambda: {name: [] for name in classes}
    )
    seen_classes: Counter = Counter()
    for candidate in candidates:
        answer = class_of(candidate)
        if answer not in classes:
            continue
        seen_classes[answer] += 1
        grouped[type_of(candidate)][answer].append(candidate)

    allowance: dict[str, int] = {}
    dropped_types: list[dict[str, Any]] = []
    for object_type, by_class in grouped.items():
        available = {name: len(rows) for name, rows in by_class.items()}
        keep = min(available.values())
        if keep == 0:
            dropped_types.append(
                {
                    "object_type": object_type,
                    "available": available,
                    "reason": "absent from at least one class",
                }
            )
            continue
        allowance[object_type] = keep

    # Trim evenly if a target size is requested: repeatedly shave one item from
    # whichever type currently contributes most, so the pool shrinks without
    # any type becoming predictive.
    if per_class_cap is not None:
        while allowance and sum(allowance.values()) > per_class_cap:
            widest = max(allowance, key=lambda name: (allowance[name], name))
            allowance[widest] -= 1
            if allowance[widest] == 0:
                del allowance[widest]

    selected: list[dict[str, Any]] = []
    for object_type, keep in sorted(allowance.items()):
        for name in classes:
            rows = sorted(grouped[object_type][name], key=order_key)
            selected.extend(dict(row) for row in rows[:keep])

    labelled = [(type_of(row), class_of(row)) for row in selected]
    oracle, chance = type_oracle_accuracy(labelled)
    kept_by_class = Counter(class_of(row) for row in selected)
    report = {
        "policy": (
            "equal count per object type in every class; types absent from any "
            "class are dropped entirely"
        ),
        "classes": list(classes),
        "input_by_class": dict(sorted(seen_classes.items())),
        "kept_by_class": dict(sorted(kept_by_class.items())),
        "types_considered": len(grouped),
        "types_kept": len(allowance),
        "types_dropped": len(dropped_types),
        "dropped_type_detail": sorted(
            dropped_types, key=lambda item: item["object_type"]
        )[:50],
        "items_discarded": sum(seen_classes.values()) - len(selected),
        "object_type_oracle_after_matching": round(oracle, 6),
        "majority_chance_after_matching": round(chance, 6),
        "type_signal_removed": abs(oracle - chance) < 1e-9,
    }
    return selected, report
