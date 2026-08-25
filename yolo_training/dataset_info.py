import os
import re
from collections import Counter, defaultdict

OUTPUT_DIR = "/media/geekscode/Developer1/Projects/gd-autopilot/yolo_training/datasets/dataset_merged_7cls"

CLASS_NAMES = [
    "mine_disc",
    "mine_rectangular_elongated",
    "mine_square",
    "mine_rectangular",
    "mine_cylinder_vert",
    "mine_directional_standing",
    "mine_round_directional_standing",
]

SPLITS = ("train", "val", "test_global", "test_real_only")
SOURCES = ("sim", "real1", "real2")

SCENARIO_RE = re.compile(r"_(spring|summer|autumn|winter)_(day|morning|evening)$", re.IGNORECASE)
MINE_RE = re.compile(r"_\d+_(obstacle_mine_\w+)_(\d+)_", re.IGNORECASE)


def scan_dataset(base_dir):
    stats = {}
    for split in SPLITS:
        lbl_dir = os.path.join(base_dir, split, "labels")
        if not os.path.isdir(lbl_dir):
            stats[split] = None
            continue

        d = {
            "total_images": 0,
            "labeled_images": 0,
            "background_images": 0,
            "total_instances": 0,
            "per_source": {s: {"images": 0, "labeled": 0, "bg": 0} for s in SOURCES},
            "per_class": Counter(),
            "per_scenario": defaultdict(lambda: {"images": 0, "instances": 0}),
        }

        for fname in sorted(os.listdir(lbl_dir)):
            if not fname.endswith(".txt"):
                continue
            d["total_images"] += 1
            stem = os.path.splitext(fname)[0]

            source = "unknown"
            for s in SOURCES:
                if stem.startswith(s + "_"):
                    source = s
                    break
            d["per_source"][source]["images"] += 1

            scenario = None
            m = SCENARIO_RE.search(stem)
            if m:
                scenario = f"{m.group(1)}_{m.group(2)}"

            with open(os.path.join(lbl_dir, fname), "r") as f:
                lines = [l.strip() for l in f if l.strip()]

            if not lines:
                d["background_images"] += 1
                d["per_source"][source]["bg"] += 1
                if scenario:
                    d["per_scenario"][scenario]["images"] += 1
            else:
                d["labeled_images"] += 1
                d["per_source"][source]["labeled"] += 1
                d["total_instances"] += len(lines)
                for line in lines:
                    d["per_class"][int(line.split()[0])] += 1
                if scenario:
                    d["per_scenario"][scenario]["images"] += 1
                    d["per_scenario"][scenario]["instances"] += len(lines)

        stats[split] = d
    return stats


def print_stats(stats):
    W = 70
    print("=" * W)
    print(f"  DATASET INFO: {OUTPUT_DIR}")
    print("=" * W)

    print(f"\n{'SPLIT OVERVIEW':=^{W}}")
    print(f"  {'Split':<16} {'Images':>7} {'Labeled':>8} {'BG':>6} {'Instances':>10}")
    print(f"  {'-'*16} {'-'*7} {'-'*8} {'-'*6} {'-'*10}")
    total_img = total_inst = 0
    for split in SPLITS:
        d = stats.get(split)
        if d is None:
            print(f"  {split:<16} (missing)")
            continue
        total_img += d["total_images"]
        total_inst += d["total_instances"]
        print(f"  {split:<16} {d['total_images']:>7d} {d['labeled_images']:>8d} "
              f"{d['background_images']:>6d} {d['total_instances']:>10d}")
    print(f"  {'-'*16} {'-'*7} {'-'*8} {'-'*6} {'-'*10}")
    print(f"  {'TOTAL':<16} {total_img:>7d} {'':>8} {'':>6} {total_inst:>10d}")

    print(f"\n{'SPLIT RATIOS':=^{W}}")
    for split in SPLITS:
        d = stats.get(split)
        if d is None:
            continue
        pct = d["total_images"] / total_img * 100 if total_img else 0
        bar = "#" * int(pct / 2)
        print(f"  {split:<16} {d['total_images']:>7d}  ({pct:5.1f}%)  {bar}")

    print(f"\n{'PER-SOURCE BREAKDOWN':=^{W}}")
    header = f"  {'Split':<16}"
    for s in SOURCES:
        header += f" {s+' img':>10} {s+' inst':>10}"
    print(header)
    print(f"  {'-'*16} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")
    for split in SPLITS:
        d = stats.get(split)
        if d is None:
            continue
        row = f"  {split:<16}"
        for s in SOURCES:
            ps = d["per_source"][s]
            row += f" {ps['images']:>10d} {'':>10}"
        print(row)
    print()
    row = f"  {'(instances)':<16}"
    for s in SOURCES:
        inst = 0
        for split in SPLITS:
            d = stats.get(split)
            if d:
                inst += sum(1 for line in _iter_labels(os.path.join(OUTPUT_DIR, split, "labels"))
                            for p in line.split() if p)

    print(f"\n  Per-source instance counts:")
    for s in SOURCES:
        inst_sum = 0
        for split in SPLITS:
            d = stats.get(split)
            if not d:
                continue
            lbl_dir = os.path.join(OUTPUT_DIR, split, "labels")
            for fname in os.listdir(lbl_dir):
                if not fname.endswith(".txt"):
                    continue
                if not fname.startswith(s + "_"):
                    continue
                with open(os.path.join(lbl_dir, fname)) as f:
                    for line in f:
                        if line.strip():
                            inst_sum += 1
        print(f"    {s:<8} {inst_sum:>7d} instances")

    print(f"\n{'PER-CLASS BREAKDOWN (instances)':=^{W}}")
    hdr = f"  {'Class':<35}"
    for split in SPLITS:
        hdr += f" {split[:6]:>8}"
    print(hdr)
    print(f"  {'-'*35}" + f" {'-'*8}" * len(SPLITS))
    for cls_id in range(len(CLASS_NAMES)):
        row = f"  {CLASS_NAMES[cls_id]:<35}"
        for split in SPLITS:
            d = stats.get(split)
            if d:
                row += f" {d['per_class'].get(cls_id, 0):>8d}"
            else:
                row += f" {'--':>8}"
        print(row)
    row = f"  {'TOTAL':<35}"
    for split in SPLITS:
        d = stats.get(split)
        if d:
            row += f" {d['total_instances']:>8d}"
        else:
            row += f" {'--':>8}"
    print(row)

    print(f"\n{'SIM SCENARIO BALANCE':=^{W}}")
    all_scenarios = set()
    for split in SPLITS:
        d = stats.get(split)
        if d:
            all_scenarios.update(d["per_scenario"].keys())
    if all_scenarios:
        hdr = f"  {'Scenario':<22}"
        for split in SPLITS:
            hdr += f" {split[:10]:>10}"
        print(hdr)
        print(f"  {'-'*22}" + f" {'-'*10}" * len(SPLITS))
        for sc in sorted(all_scenarios):
            row = f"  {sc:<22}"
            for split in SPLITS:
                d = stats.get(split)
                if d and sc in d["per_scenario"]:
                    row += f" {d['per_scenario'][sc]['images']:>10d}"
                else:
                    row += f" {'--':>10}"
            print(row)
        # total row
        row = f"  {'TOTAL':<22}"
        for split in SPLITS:
            d = stats.get(split)
            if d:
                total = sum(v["images"] for v in d["per_scenario"].values())
                row += f" {total:>10d}"
            else:
                row += f" {'--':>10}"
        print(row)

    print(f"\n{'CLASS IMBALANCE':=^{W}}")
    train_d = stats.get("train")
    if train_d and train_d["per_class"]:
        counts = train_d["per_class"]
        max_cls = counts.most_common(1)[0]
        min_cls = counts.most_common()[-1]
        ratio = max_cls[1] / min_cls[1] if min_cls[1] > 0 else float("inf")
        print(f"  Most frequent:  {CLASS_NAMES[max_cls[0]]} ({max_cls[1]})")
        print(f"  Least frequent: {CLASS_NAMES[min_cls[0]]} ({min_cls[1]})")
        print(f"  Ratio:          {ratio:.1f}x")

    print()


def _iter_labels(lbl_dir):
    if not os.path.isdir(lbl_dir):
        return
    for fname in sorted(os.listdir(lbl_dir)):
        if fname.endswith(".txt"):
            with open(os.path.join(lbl_dir, fname)) as f:
                for line in f:
                    yield line.strip()


if __name__ == "__main__":
    stats = scan_dataset(OUTPUT_DIR)
    print_stats(stats)
