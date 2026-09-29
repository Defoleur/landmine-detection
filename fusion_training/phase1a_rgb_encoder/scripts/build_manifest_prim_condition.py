#!/usr/bin/env python3
"""Build a manifest grouped by Prim-path identity and stratified by condition.

--mode prim (default): group records sharing visible_mines[].prim_path, split
  to cover every season+daytime condition.  Prevents Prim leakage across splits.
--mode condition: independent records, no prim-path grouping, split balancing
  both season+daytime conditions and mine types (exact_class or mine_type).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from phase1a_rgb_encoder.data import dataset_report, discover_records, write_manifest


def prim_paths(record: dict) -> set[str]:
    value = record.get("visible_mine_prim_paths", "[]")
    if isinstance(value, list):
        return set(value)
    try:
        return set(json.loads(value or "[]"))
    except (TypeError, json.JSONDecodeError):
        return set()


def extract_mine_types(
    record: dict, dataset_root: Path, mine_type_field: str = "exact_class"
) -> set[str]:
    """Read mine type labels from the record's metadata JSON."""
    metadata_path = dataset_root / "metadata" / f"{record['acquisition_id']}.json"
    if not metadata_path.exists():
        return {"unknown"}
    metadata = json.loads(metadata_path.read_text())
    types = {
        mine.get(mine_type_field)
        for mine in metadata.get("visible_mines", [])
        if mine.get(mine_type_field)
    }
    return types or {"unknown"}


def make_groups(records: list[dict]) -> list[list[int]]:
    """Connect records sharing a Prim path; keep path-less records independent."""
    parent = list(range(len(records)))

    def find(value: int) -> int:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(left: int, right: int) -> None:
        left, right = find(left), find(right)
        if left != right:
            parent[right] = left

    path_owner: dict[str, int] = {}
    for index, record in enumerate(records):
        for path in prim_paths(record):
            if path in path_owner:
                union(index, path_owner[path])
            else:
                path_owner[path] = index

    groups: dict[int, list[int]] = defaultdict(list)
    for index in range(len(records)):
        groups[find(index)].append(index)
    return list(groups.values())


def assign_splits(
    records: list[dict], seed: int, ratios: tuple[float, float, float]
) -> list[dict]:
    groups = make_groups(records)
    if len(groups) < 3:
        raise ValueError("At least three independent Prim-path groups are required")

    split_names = ("train", "val", "test")
    conditions = sorted(
        {record.get("configuration_id", "unknown") for record in records}
    )
    target_sizes = [len(records) * ratio for ratio in ratios]
    condition_totals = Counter(
        record.get("configuration_id", "unknown") for record in records
    )
    target_conditions = {
        condition: [condition_totals[condition] * ratio for ratio in ratios]
        for condition in conditions
    }

    units = []
    for members in groups:
        unit_conditions = Counter(
            records[index].get("configuration_id", "unknown") for index in members
        )
        paths = sorted(
            {path for index in members for path in prim_paths(records[index])}
        )
        group_key = "|".join(paths) or records[members[0]]["acquisition_id"]
        group_id = f"prim_{hashlib.sha1(group_key.encode()).hexdigest()[:12]}"
        units.append((members, unit_conditions, group_id))

    units_per_condition = {
        condition: sum(
            1
            for _, unit_conditions, _ in units
            if unit_conditions.get(condition, 0) > 0
        )
        for condition in conditions
    }
    coverable = {
        condition for condition in conditions if units_per_condition[condition] >= 3
    }

    best = None
    for attempt in range(200):
        rng = random.Random(seed + attempt)
        order = list(range(len(units)))
        rng.shuffle(order)
        order.sort(key=lambda index: len(units[index][0]), reverse=True)
        totals = [0, 0, 0]
        condition_counts = [Counter(), Counter(), Counter()]
        assignments: dict[int, int] = {}

        for unit_index in order:
            members, unit_conditions, _ = units[unit_index]
            scores = []
            for split in range(3):
                size = totals[split] + len(members)
                size_ratio = size / max(target_sizes[split], 1)
                condition_error = 0.0
                for condition in conditions:
                    count = (
                        condition_counts[split][condition] + unit_conditions[condition]
                    )
                    target = target_conditions[condition][split]
                    condition_error += ((count - target) / max(target, 1)) ** 2
                # Keep the requested overall split ratio primary. Conditions
                # are tie-breaking guidance, without recreating giant groups.
                scores.append(size_ratio + condition_error * 1e-4 + rng.random() * 1e-6)
            split = min(range(3), key=lambda index: scores[index])
            assignments[unit_index] = split
            totals[split] += len(members)
            condition_counts[split].update(unit_conditions)

        if any(
            condition_counts[split][condition] == 0
            for split in range(3)
            for condition in coverable
        ):
            continue
        size_objective = sum(
            ((totals[index] - target_sizes[index]) / max(target_sizes[index], 1)) ** 2
            for index in range(3)
        )
        condition_objective = sum(
            (
                (
                    condition_counts[index][condition]
                    - target_conditions[condition][index]
                )
                / max(target_conditions[condition][index], 1)
            )
            ** 2
            for index in range(3)
            for condition in conditions
        )
        objective = size_objective + condition_objective * 1e-4
        if best is None or objective < best[0]:
            best = (objective, assignments, condition_counts)

    if best is None:
        raise ValueError(
            "Could not create condition-stratified train/val/test splits. "
            "Check that every condition with >= 3 independent groups can be balanced."
        )

    _, assignments, best_condition_counts = best
    for unit_index, split in assignments.items():
        members, _, group_id = units[unit_index]
        for index in members:
            records[index]["split"] = split_names[split]
            records[index]["split_group_id"] = group_id
    conditions_per_split = {
        condition: {
            name: best_condition_counts[split_index].get(condition, 0)
            for split_index, name in enumerate(split_names)
        }
        for condition in conditions
    }
    return records, {
        "units_per_condition": units_per_condition,
        "conditions_per_split": conditions_per_split,
    }


def assign_condition_splits(
    records: list[dict],
    seed: int,
    ratios: tuple[float, float, float],
    dataset_root: Path,
    mine_type_field: str = "exact_class",
) -> list[dict]:
    """Split by condition first, then balance mine types within each split."""
    if not records:
        return records, {}

    split_names = ("train", "val", "test")
    conditions = sorted(
        {record.get("configuration_id", "unknown") for record in records}
    )
    record_mine_types = [
        extract_mine_types(record, dataset_root, mine_type_field) for record in records
    ]
    all_mine_types = sorted({mt for types in record_mine_types for mt in types})

    condition_indices = {
        condition: [
            index
            for index, record in enumerate(records)
            if record.get("configuration_id", "unknown") == condition
        ]
        for condition in conditions
    }

    def split_counts(total: int) -> list[int]:
        counts = [int(total * ratio) for ratio in ratios]
        for split in sorted(
            range(3), key=lambda index: -(total * ratios[index] - counts[index])
        ):
            if sum(counts) >= total:
                break
            counts[split] += 1
        return counts

    best = None
    for attempt in range(200):
        rng = random.Random(seed + attempt)
        mt_counts = [Counter(), Counter(), Counter()]
        condition_mt_counts = {
            condition: [Counter(), Counter(), Counter()] for condition in conditions
        }
        assignments: dict[int, int] = {}

        for condition in conditions:
            indices = list(condition_indices[condition])
            rng.shuffle(indices)
            remaining = split_counts(len(indices))
            condition_totals_by_mine = Counter(
                mine_type for index in indices for mine_type in record_mine_types[index]
            )
            targets = {
                mine_type: [
                    condition_totals_by_mine[mine_type] * ratio for ratio in ratios
                ]
                for mine_type in condition_totals_by_mine
            }
            for index in indices:
                mine_types = record_mine_types[index]
                scores = []
                for split in range(3):
                    if not remaining[split]:
                        scores.append(float("inf"))
                        continue
                    score = sum(
                        (
                            (condition_mt_counts[condition][split][mine_type] + 1)
                            / max(targets[mine_type][split], 1)
                        )
                        ** 2
                        for mine_type in mine_types
                    ) / max(len(mine_types), 1)
                    scores.append(score + rng.random() * 1e-6)
                split = min(range(3), key=lambda value: scores[value])
                assignments[index] = split
                remaining[split] -= 1
                condition_mt_counts[condition][split].update(mine_types)
                mt_counts[split].update(mine_types)

        objective = sum(
            (
                (
                    condition_mt_counts[condition][split][mine_type]
                    - sum(
                        1
                        for index in condition_indices[condition]
                        if mine_type in record_mine_types[index]
                    )
                    * ratios[split]
                )
                / max(
                    1,
                    sum(
                        1
                        for index in condition_indices[condition]
                        if mine_type in record_mine_types[index]
                    )
                    * ratios[split],
                )
            )
            ** 2
            for condition in conditions
            for mine_type in all_mine_types
            for split in range(3)
        )
        if best is None or objective < best[0]:
            best = (objective, assignments, condition_mt_counts, mt_counts)

    if best is None:
        raise ValueError(
            "Could not create condition+mine-type stratified train/val/test splits"
        )

    _, assignments, best_condition_mt_counts, best_mt_counts = best
    for idx, split in assignments.items():
        records[idx]["split"] = split_names[split]
        records[idx]["split_group_id"] = f"cond_{split_names[split]}"
    return records, {
        "conditions_per_split": {
            condition: {
                name: sum(
                    1
                    for index, split in assignments.items()
                    if records[index].get("configuration_id", "unknown") == condition
                    and split == split_index
                )
                for split_index, name in enumerate(split_names)
            }
            for condition in conditions
        },
        "mine_type_per_split": {
            name: dict(best_mt_counts[i]) for i, name in enumerate(split_names)
        },
        "mine_type_per_condition_split": {
            condition: {
                name: dict(best_condition_mt_counts[condition][split_index])
                for split_index, name in enumerate(split_names)
            }
            for condition in conditions
        },
    }


def _report_split_diagnostics(
    report: dict, split_diagnostics: dict, records: list[dict]
) -> None:
    """Add common split diagnostics to the report dict."""
    split_names = ("train", "val", "test")
    conditions_not_in_all_splits = {}
    for condition, per_split in split_diagnostics["conditions_per_split"].items():
        present = [name for name in split_names if per_split.get(name, 0) > 0]
        if len(present) < len(split_names):
            conditions_not_in_all_splits[condition] = {
                "record_count": sum(per_split.values()),
                "independent_groups": split_diagnostics.get(
                    "units_per_condition", {}
                ).get(condition),
                "present_in": present,
            }
    if conditions_not_in_all_splits:
        report["conditions_not_in_all_splits"] = conditions_not_in_all_splits


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--mode",
        choices=("prim", "condition"),
        default="prim",
        help="prim: group by Prim path, stratify by season+daytime (default). "
        "condition: independent records, stratify by season+daytime + mine type.",
    )
    parser.add_argument(
        "--mine-type-field",
        default="exact_class",
        help="Metadata field to extract per-mine labels in condition mode (default: exact_class)",
    )
    args = parser.parse_args()

    records = discover_records(args.dataset_root)

    if args.mode == "prim":
        records, split_diagnostics = assign_splits(records, args.seed, (0.8, 0.1, 0.1))
    else:
        records, split_diagnostics = assign_condition_splits(
            records, args.seed, (0.8, 0.1, 0.1), args.dataset_root, args.mine_type_field
        )

    write_manifest(records, args.output)
    report = dataset_report(records)
    report.update(
        {
            "grouping": (
                "shared visible_mines[].prim_path only"
                if args.mode == "prim"
                else f"independent records, stratified by {args.mine_type_field}"
            ),
            "stratification": (
                "configuration_id (season + daytime)"
                if args.mode == "prim"
                else f"configuration_id (season + daytime) + {args.mine_type_field}"
            ),
            "spatial_block_grouping": False,
            "seed": args.seed,
            "source_dataset": str(args.dataset_root.resolve()),
        }
    )
    if args.mode == "condition":
        report["prim_paths_crossing_splits"] = (
            "expected; records are split individually, not grouped by prim_path"
        )
    _report_split_diagnostics(report, split_diagnostics, records)

    if "mine_type_per_split" in split_diagnostics:
        report["mine_type_per_split"] = split_diagnostics["mine_type_per_split"]
    if "mine_type_per_condition_split" in split_diagnostics:
        report["mine_type_per_condition_split"] = split_diagnostics[
            "mine_type_per_condition_split"
        ]

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
