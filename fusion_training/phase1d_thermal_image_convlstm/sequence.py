"""Fixed twelve-condition manifests using processed thermal images."""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

from phase1d_thermal_convlstm.sequence import (
    CONDITIONS,
    _split_points,
    discover_target_masks,
)

CONDITION_RE = re.compile(r"_(spring|summer|autumn|winter)_(morning|day|evening)$")
POINT_RE = re.compile(r"_(p\d{2})_obstacle_mine_(.+)$")


def discover_point_frames(dataset_root, processed_root):
    source = Path(dataset_root).resolve()
    image_root = Path(processed_root).resolve() / "images"
    records = []
    for image_path in sorted(image_root.glob("*.png")):
        condition_match = CONDITION_RE.search(image_path.stem)
        point_match = POINT_RE.search(
            image_path.stem[: condition_match.start()]
            if condition_match
            else image_path.stem
        )
        if not condition_match or not point_match:
            continue
        stem = image_path.stem
        mask_path = source / "masks" / f"{stem}.png"
        metadata_path = source / "metadata" / f"{stem}.json"
        metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
        records.append(
            {
                "frame_id": stem,
                "mine_id": point_match.group(2),
                "point_number": int(point_match.group(1)[1:]),
                "condition": f"{condition_match.group(1)}_{condition_match.group(2)}",
                "image_path": str(image_path),
                "target_path": str(mask_path) if mask_path.exists() else "",
                "metadata_path": str(metadata_path),
                "position": metadata.get("position"),
                "point_index": metadata.get("point_index"),
                "points_per_mine": metadata.get("points_per_mine"),
                "trajectory": metadata.get("trajectory"),
            }
        )
    if not records:
        raise ValueError(f"No processed point frames found under {image_root}")
    return records


def build_sequence_manifest(
    dataset_root,
    processed_root,
    seed=42,
    require_complete=True,
    target_mask_dataset=None,
    merge_mine_variants=False,
    split_ratios=None,
):
    frames = discover_point_frames(dataset_root, processed_root)
    external = discover_target_masks(target_mask_dataset) if target_mask_dataset else {}
    groups = defaultdict(dict)
    for frame in frames:
        groups[(frame["mine_id"], frame["point_number"])][frame["condition"]] = frame
    splits = _split_points(
        groups,
        seed,
        merge_mine_variants=merge_mine_variants,
        split_ratios=split_ratios,
    )
    if external and set(external) != set(groups):
        missing = sorted(set(groups) - set(external))
        extra = sorted(set(external) - set(groups))
        raise ValueError(
            "External target identities do not match source mine/point identities: "
            f"missing={missing[:3]} extra={extra[:3]}"
        )
    metadata_mismatches = []
    if external:
        for key, target in external.items():
            source = groups[key].get("autumn_day") or next(iter(groups[key].values()))
            mismatch = {
                field: (source.get(field), target.get(field))
                for field in ("point_index", "points_per_mine", "trajectory")
                if source.get(field) != target.get(field)
            }
            if mismatch:
                metadata_mismatches.append({"key": key, "fields": mismatch})
        if metadata_mismatches:
            raise ValueError(
                f"Target point metadata mismatch: {metadata_mismatches[:3]}"
            )
    sequences, rejected = [], []
    for (mine_id, point_number), by_condition in sorted(groups.items()):
        missing = sorted(set(CONDITIONS) - set(by_condition))
        if missing and require_complete:
            rejected.append(
                {"mine_id": mine_id, "point_number": point_number, "missing": missing}
            )
            continue
        frames_ordered = [by_condition[c] for c in CONDITIONS if c in by_condition]
        target = (
            external.get((mine_id, point_number), {}).get("target_path")
            if external
            else None
        )
        sequences.append(
            {
                "sequence_id": f"{mine_id}_p{point_number:02d}",
                "mine_id": mine_id,
                "point_number": point_number,
                "split": splits[(mine_id, point_number)],
                "conditions": [f["condition"] for f in frames_ordered],
                "frames": frames_ordered,
                "target_path": target or frames_ordered[-1]["target_path"],
            }
        )
    report = {
        "source_frames": len(frames),
        "sequences": len(sequences),
        "rejected": rejected,
        "splits": {
            s: sum(x["split"] == s for x in sequences) for s in ("train", "val", "test")
        },
        "conditions": list(CONDITIONS),
        "seed": seed,
        "dataset_root": str(Path(dataset_root).resolve()),
        "processed_root": str(Path(processed_root).resolve()),
        "merge_mine_variants": merge_mine_variants,
    }
    if split_ratios:
        report["split_ratios"] = list(split_ratios)
    if target_mask_dataset:
        report["target_mask_dataset"] = str(Path(target_mask_dataset).resolve())
        report["target_mask_points"] = len(external)
        report["target_metadata_mismatches"] = len(metadata_mismatches)
    return sequences, report


def write_manifest(sequences, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(sequences, indent=2) + "\n")


def read_manifest(path):
    return json.loads(Path(path).read_text())
