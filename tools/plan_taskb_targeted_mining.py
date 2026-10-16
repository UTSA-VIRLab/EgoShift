#!/usr/bin/env python3
"""Plan a superseding Task-B targeted-mining contract from regenerated roles.

Recomputes per-cell reservoir deficits with the same library calls as the S05
preflight (``finite_label_quotas``, ``load_candidates``,
``_assignments_for_role``), then mints a new mining config that supersedes the
base contract with role-correct targets. Fails closed if any deficit falls in
a cell the base contract does not mine.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from egoshift.hashing import sha256_file  # noqa: E402
from egoshift.split_roles import require_complete_split_roles  # noqa: E402
from egoshift.taskb_candidate_r2 import (  # noqa: E402
    ROLES,
    SourceSpec,
    _assignments_for_role,
    finite_label_quotas,
    load_candidates,
    validate_contract,
)
from egoshift.taskb_geometry import (  # noqa: E402
    canonical_zone,
    post_action_bearing_degrees,
)

ZONE_BOUNDS = {
    "in front": (-45.0, 45.0),
    "to the right": (45.0, 135.0),
    "behind": (135.0, 225.0),
    "to the left": (225.0, 315.0),
}


def feasible_centre_band(
    rotation: int,
    geometry_turn: str,
    answer: str,
    half_width: float,
    margin: float,
    step: float = 0.05,
) -> tuple[float, float] | None:
    """Centre bearings a point target may occupy under the frozen gates.

    Applies the same two gates the miner applies: the source-cone half-width
    and the whole-box post-action zone margin (evaluated for a point target,
    which is the upper bound any real box can achieve). Nothing here relaxes a
    gate; it only reports where the gates can be satisfied so the search can
    aim there instead of at an infeasible bearing.
    """

    low, high = ZONE_BOUNDS[answer]
    inside: list[float] = []
    bearing = -half_width
    while bearing <= half_width:
        if canonical_zone(post_action_bearing_degrees(bearing, rotation, geometry_turn)) == answer:
            post = post_action_bearing_degrees(bearing, rotation, geometry_turn) % 360.0
            nearest = min(
                abs(((post - bound + 180.0) % 360.0) - 180.0) for bound in (low, high)
            )
            if nearest >= margin:
                inside.append(bearing)
        bearing += step
    if not inside:
        return None
    return min(inside), max(inside)


def padded_target(shortfall: int) -> int:
    """Deterministic padding: shortfall plus max(2, 15%) rounded up to even."""
    if shortfall <= 0:
        return 0
    padded = shortfall + max(2, math.ceil(0.15 * shortfall))
    return padded + (padded % 2)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--split-roles", required=True, type=Path)
    parser.add_argument("--reservoir-index", required=True, type=Path)
    parser.add_argument("--reservoir-geometry-report", required=True, type=Path)
    parser.add_argument("--reservoir-asset-root", required=True, type=Path)
    parser.add_argument("--reservoir-asset-trust-root", required=True, type=Path)
    parser.add_argument("--base-config", required=True, type=Path)
    parser.add_argument("--targeted-index", action="append", default=[], type=Path)
    parser.add_argument("--targeted-geometry-report", action="append", default=[], type=Path)
    parser.add_argument("--targeted-merge-report", action="append", default=[], type=Path)
    parser.add_argument("--targeted-asset-root", action="append", default=[], type=Path)
    parser.add_argument("--targeted-asset-trust-root", action="append", default=[], type=Path)
    parser.add_argument(
        "--auto-aim",
        action="store_true",
        help="Re-aim each cell at the centre of its gate-feasible centre band.",
    )
    parser.add_argument("--jitter-count", type=int, default=3)
    parser.add_argument("--narrow-band-degrees", type=float, default=20.0)
    parser.add_argument("--narrow-target-bbox-width-px", type=float, default=24.0)
    parser.add_argument("--maximum-jitter-degrees", type=float, default=4.0)
    parser.add_argument("--max-positions", type=int, default=None)
    parser.add_argument("--max-attempts", type=int, default=None)
    parser.add_argument("--max-objects", type=int, default=None)
    parser.add_argument("--only-deficient-cells", action="store_true")
    parser.add_argument("--mining-id", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists() or args.report.exists():
        raise ValueError("refusing to overwrite an existing plan/report")

    config = json.loads(args.config.read_text(encoding="utf-8"))
    validate_contract(config)
    roles = json.loads(args.split_roles.read_text(encoding="utf-8"))
    exposure_binding = require_complete_split_roles(roles)

    targeted_lists = (
        args.targeted_index,
        args.targeted_geometry_report,
        args.targeted_merge_report,
        args.targeted_asset_root,
        args.targeted_asset_trust_root,
    )
    if len({len(item) for item in targeted_lists}) != 1:
        raise SystemExit("targeted source flags must be supplied the same number of times")
    specs = [
        SourceSpec(
            "reservoir_geometry_b665868",
            "reservoir",
            args.reservoir_index.resolve(),
            args.reservoir_geometry_report.resolve(),
            args.reservoir_asset_root,
            args.reservoir_asset_trust_root,
        )
    ]
    for position, (index, report, merge_report, asset_root, trust_root) in enumerate(
        zip(*targeted_lists), start=1
    ):
        specs.append(
            SourceSpec(
                f"targeted_prior_{position}",
                "targeted",
                index.resolve(),
                report.resolve(),
                asset_root,
                trust_root,
                merge_report.resolve(),
            )
        )
    candidates, source_audit = load_candidates(
        source_specs=specs, roles_payload=roles, config=config
    )

    seed = str(config["seed"])
    half_width = float(config["geometry"]["source_bearing_half_width_degrees"])
    base = json.loads(args.base_config.read_text(encoding="utf-8"))
    base_cells = {
        (int(c["rotation_degrees"]), str(c["turn"]), str(c["answer"]))
        for c in base["cells"]
    }

    presented_deficits: dict[str, dict[str, int]] = {}
    consolidated_180: dict[str, dict[str, int]] = {}
    targets: dict[str, dict[str, int]] = {role: {} for role in ROLES}
    uncovered: list[str] = []
    raw_minimum = 0
    for role in ROLES:
        quotas = finite_label_quotas(
            role=role,
            items_per_action_cell=int(config["splits"][role]["items_per_action_cell"]),
            seed=seed,
            half_width=half_width,
            items_by_rotation=config["splits"][role].get(
                "items_per_action_cell_by_rotation"
            ),
        )
        assignments = _assignments_for_role(candidates, role, quotas)
        capacity = Counter(item.stratum for item in assignments)
        for key, required in sorted(quotas.items()):
            if key[0] == 180:
                continue
            available = capacity[key]
            shortfall = required - available
            if shortfall > 0:
                name = f"{role}:{key[0]}:{key[1]}:{key[2]}"
                presented_deficits[name] = {
                    "available": available,
                    "required": required,
                    "shortfall": shortfall,
                }
                raw_minimum += shortfall
                if key not in base_cells:
                    uncovered.append(name)
                else:
                    targets[role][f"{key[0]}:{key[1]}:{key[2]}"] = padded_target(
                        shortfall
                    )
        for answer in ("to the right", "behind", "to the left"):
            required = sum(quotas[(180, turn, answer)] for turn in ("left", "right"))
            available = sum(
                1
                for row in candidates
                if row["_role"] == role
                and int(row["rotation_degrees"]) == 180
                and row["answer"] == answer
            )
            shortfall = max(0, required - available)
            name = f"{role}:180:any:{answer}"
            consolidated_180[name] = {
                "available_unique_source_rows": available,
                "required_across_two_presented_turns": required,
                "shortfall": shortfall,
            }
            raw_minimum += shortfall
            if shortfall > 0:
                if (180, "any", answer) not in base_cells:
                    uncovered.append(name)
                else:
                    targets[role][f"180:any:{answer}"] = padded_target(shortfall)
    if uncovered:
        raise SystemExit(
            f"deficit cells not covered by base mining cells: {uncovered}"
        )
    for role in ROLES:
        for cell in base["cells"]:
            cid = f"{cell['rotation_degrees']}:{cell['turn']}:{cell['answer']}"
            targets[role].setdefault(cid, 0)

    requested = sum(v for role in targets.values() for v in role.values())
    minimum_before_caps = raw_minimum + 10  # frozen joint-cap reserve from r2b
    plan = dict(base)

    deficient_cells = {
        cid
        for role_targets in targets.values()
        for cid, value in role_targets.items()
        if value
    }
    if args.only_deficient_cells:
        plan["cells"] = [
            cell
            for cell in base["cells"]
            if f"{cell['rotation_degrees']}:{cell['turn']}:{cell['answer']}"
            in deficient_cells
        ]
        kept = {
            f"{cell['rotation_degrees']}:{cell['turn']}:{cell['answer']}"
            for cell in plan["cells"]
        }
        targets = {
            role: {cid: value for cid, value in role_targets.items() if cid in kept}
            for role, role_targets in targets.items()
        }
        requested = sum(v for role_targets in targets.values() for v in role_targets.values())

    aim_report: dict[str, Any] = {}
    if args.auto_aim:
        margin = float(base["gates"]["minimum_whole_box_post_zone_margin_degrees"])
        cone = float(base["gates"]["source_bearing_half_width_degrees"])
        render = base["render"]
        half_fov = math.degrees(
            math.atan(
                math.tan(math.radians(float(render["vertical_fov_degrees"]) / 2.0))
                * float(render["width"])
                / float(render["height"])
            )
        )
        degrees_per_pixel = 2.0 * half_fov / float(render["width"])
        aimed_cells = []
        for cell in plan["cells"]:
            cell = dict(cell)
            rotation = int(cell["rotation_degrees"])
            geometry_turn = "left" if rotation == 180 else str(cell["turn"])
            band = feasible_centre_band(
                rotation, geometry_turn, str(cell["answer"]), cone, margin
            )
            cid = f"{rotation}:{cell['turn']}:{cell['answer']}"
            if band is None:
                raise SystemExit(
                    f"cell {cid} has an empty gate-feasible centre band; "
                    "mining it cannot succeed and the quota must be re-frozen"
                )
            low, high = band
            width = high - low
            narrow = width <= float(args.narrow_band_degrees)
            aim_low, aim_high = low, high
            if narrow:
                # Narrow band: only targets whose horizontal extent is small
                # enough can satisfy the frozen whole-box margin and stay
                # clear of the frame edge.  Aim the camera distance at the
                # bbox width that keeps both satisfiable, then inset the band
                # by that target's own angular half-extent -- a point-target
                # band overstates what a real box can occupy.
                cell["target_bbox_width_pixels"] = float(
                    args.narrow_target_bbox_width_px
                )
                half_extent = (
                    float(args.narrow_target_bbox_width_px) * degrees_per_pixel / 2.0
                )
                edge_allowance = (
                    float(base["gates"]["minimum_mask_edge_margin_pixels"])
                    * degrees_per_pixel
                )
                if low >= 0.0:
                    aim_low = low + half_extent
                    aim_high = min(high, half_fov - half_extent - edge_allowance)
                else:
                    aim_low = max(low, -(half_fov - half_extent - edge_allowance))
                    aim_high = high - half_extent
                if aim_low > aim_high:
                    raise SystemExit(
                        f"cell {cid}: no centre remains after insetting the "
                        f"feasible band by a {args.narrow_target_bbox_width_px:.0f}px "
                        "target; the quota must be re-frozen rather than mined"
                    )
            centre = round((aim_low + aim_high) / 2.0, 3)
            count = max(1, int(args.jitter_count))
            if count == 1:
                offsets = [0.0]
            else:
                # Span the usable sub-band, but never push jitters onto the
                # margin boundary of a wide band, where any real box fails.
                span = min((aim_high - aim_low) / 2.0, float(args.maximum_jitter_degrees))
                offsets = [
                    round(-span + (2.0 * span) * index / (count - 1), 3)
                    for index in range(count)
                ]
            previous_aim = float(cell["desired_source_bearing_degrees"])
            previous_in_band = low <= previous_aim <= high
            cell["desired_source_bearing_degrees"] = centre
            cell["bearing_jitters_degrees"] = offsets
            aim_report[cid] = {
                "feasible_centre_band": [round(low, 3), round(high, 3)],
                "band_width_degrees": round(width, 3),
                "usable_aim_band": [round(aim_low, 3), round(aim_high, 3)],
                "previous_desired": previous_aim,
                "previous_aim_inside_band": previous_in_band,
                "new_desired": centre,
                "jitters": offsets,
                "target_bbox_width_pixels": cell.get("target_bbox_width_pixels"),
            }
            aimed_cells.append(cell)
        plan["cells"] = aimed_cells

    for knob, value in (
        ("maximum_positions_per_object_cell", args.max_positions),
        ("maximum_render_attempts_per_scene", args.max_attempts),
        ("maximum_objects_per_scene", args.max_objects),
    ):
        if value is not None:
            plan[knob] = int(value)
    plan["search_effort_note"] = (
        "Search effort and aiming only. Every gate in `gates`, the per-component "
        "and per-split caps, and the finite-target contract are byte-identical to "
        "the base configuration; only where and how hard the miner looks changed."
    )
    plan["mining_id"] = args.mining_id
    plan["supersedes_operationally"] = str(args.base_config)
    plan["supersession_reason"] = (
        "Roles were regenerated from the exhaustive reviewed historical-"
        "exposure ledger; every r2b per-cell target derived from the "
        "invalidated three-file ledger is void. Targets here are recomputed "
        "from the corrected reservoir under the new roles with the same "
        "padding policy (shortfall + max(2, 15%) rounded up to even)."
    )
    plan["target_rows_per_role_and_cell"] = targets
    plan["quota_basis"] = {
        "candidate_items_per_presented_action": {
            role: int(config["splits"][role]["items_per_action_cell"])
            for role in ROLES
        },
        "qualified_reservoir_source": str(args.reservoir_index),
        "qualified_reservoir_sha256": sha256_file(args.reservoir_index),
        "minimum_fresh_rows_before_caps": minimum_before_caps,
        "requested_rows_before_merge_and_selection": requested,
        "buffer_rows": requested - minimum_before_caps,
        "split_roles_binding": {
            "path": str(args.split_roles),
            "sha256": sha256_file(args.split_roles),
            "exposure_manifest_sha256": exposure_binding["manifest_sha256"],
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(plan, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    report = {
        "audit": "taskb_targeted_mining_plan",
        "config": {"path": str(args.config), "sha256": sha256_file(args.config)},
        "split_roles": {
            "path": str(args.split_roles),
            "sha256": sha256_file(args.split_roles),
        },
        "source_audit": source_audit,
        "auto_aim": aim_report,
        "presented_non180_deficit_cells": presented_deficits,
        "consolidated_180_unique_source_deficits": consolidated_180,
        "raw_minimum_fresh_rows": raw_minimum,
        "minimum_fresh_rows_before_caps": minimum_before_caps,
        "requested_rows": requested,
        "targets": targets,
        "output_config": {
            "path": str(args.output),
            "sha256": sha256_file(args.output),
        },
        "pass": True,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False, default=str)
        + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "raw_minimum_fresh_rows": raw_minimum,
                "requested_rows": requested,
                "targets": targets,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
