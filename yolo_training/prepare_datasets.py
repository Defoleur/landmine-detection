import os
import random
import re
import shutil
from collections import Counter, defaultdict

SEED = 42

SIM_DIRS = [
    "/media/geekscode/Developer1/Projects/gd-autopilot/datasets/dataset_auto_last_20260814_180837",
    "/media/geekscode/Developer1/Projects/gd-autopilot/datasets/dataset_auto_last_20260814_180837_empty",
]
REAL1_DIR = "/media/geekscode/Developer1/Projects/gd-autopilot/yolo_training/datasets/lwir_real_all"
REAL2_DIR = "/media/geekscode/Developer1/Projects/gd-autopilot/yolo_training/datasets/landmine.v5i.yolo26"
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

SIM_ID_MAP   = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 6: 5, 7: 6}
REAL1_ID_MAP = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 6: 5, 7: 6}
REAL2_ID_MAP = {0: 0}

SIM_SPLIT   = (0.80, 0.10, 0.10)
REAL_SPLIT  = (0.70, 0.15, 0.15)

EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")
SPLITS = ("train", "val", "test_global", "test_real_only")

SCENARIO_RE = re.compile(r"_(spring|summer|autumn|winter)_(day|morning|evening)\.png$", re.IGNORECASE)
MINE_RE = re.compile(r"_\d+_(obstacle_mine_\w+)_(\d+)_", re.IGNORECASE)


def parse_sim_filename(fname):
    m_sc = SCENARIO_RE.search(fname)
    scenario = f"{m_sc.group(1)}_{m_sc.group(2)}" if m_sc else None
    m_mine = MINE_RE.search(fname)
    mine_key = f"{m_mine.group(1)}_{m_mine.group(2)}" if m_mine else None
    return mine_key, scenario


def quad_to_hbb(coords):
    xs = coords[0::2]
    ys = coords[1::2]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    return (x_min + x_max) / 2.0, (y_min + y_max) / 2.0, x_max - x_min, y_max - y_min


def convert_label(src_path, dst_path, id_map):
    lines_out = []
    with open(src_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            try:
                old_id = int(parts[0])
            except ValueError:
                continue
            new_id = id_map.get(old_id)
            if new_id is None:
                continue
            coords = [float(p) for p in parts[1:]]
            if len(coords) == 8:
                x_c, y_c, w, h = quad_to_hbb(coords)
            elif len(coords) == 4:
                x_c, y_c, w, h = coords
            else:
                continue
            if w <= 0 or h <= 0:
                continue
            lines_out.append(
                f"{new_id} {max(0, min(1, x_c)):.6f} {max(0, min(1, y_c)):.6f} "
                f"{max(0, min(1, w)):.6f} {max(0, min(1, h)):.6f}"
            )
    with open(dst_path, "w") as f:
        if lines_out:
            f.write("\n".join(lines_out) + "\n")


def safe_link(src, dst):
    if os.path.exists(dst):
        if os.path.samefile(src, dst):
            return
        os.remove(dst)
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def collect_images(src_dirs):
    collected = []
    for src in src_dirs:
        if not os.path.isdir(src):
            print(f"  WARN: source not found: {src}")
            continue
        flat_images = os.path.join(src, "images")
        if os.path.isdir(flat_images):
            images_dir = flat_images
            labels_dir = os.path.join(src, "labels")
            for fname in sorted(os.listdir(images_dir)):
                if fname.lower().endswith(EXTENSIONS):
                    src_img = os.path.join(images_dir, fname)
                    stem = os.path.splitext(fname)[0]
                    lbl = os.path.join(labels_dir, stem + ".txt")
                    collected.append((fname, src_img, lbl if os.path.exists(lbl) else None))
        else:
            for sub in ("train", "valid", "test"):
                sub_img = os.path.join(src, sub, "images")
                if not os.path.isdir(sub_img):
                    continue
                sub_lbl = os.path.join(src, sub, "labels")
                for fname in sorted(os.listdir(sub_img)):
                    if fname.lower().endswith(EXTENSIONS):
                        src_img = os.path.join(sub_img, fname)
                        stem = os.path.splitext(fname)[0]
                        lbl = os.path.join(sub_lbl, stem + ".txt")
                        collected.append((fname, src_img, lbl if os.path.exists(lbl) else None))
    return collected


def greedy_assign_groups(groups, scenario_totals, ratios):
    train_val_test = ("train", "val", "test_global")
    splits = {s: [] for s in train_val_test}
    split_scenario = {s: Counter() for s in train_val_test}
    target = {s: Counter() for s in train_val_test}
    for sc, total in scenario_totals.items():
        for ratio, split in zip(ratios, train_val_test):
            target[split][sc] = total * ratio

    for _, items in groups:
        scenario_vec = Counter()
        for _, _, _, sc in items:
            scenario_vec[sc] += 1

        best_split = None
        best_ratio = float("inf")
        for split in train_val_test:
            ratio = 0
            for sc in scenario_vec:
                t = target[split][sc]
                current = split_scenario[split][sc]
                if t > 0:
                    ratio += current / t
                else:
                    ratio += float("inf")
            if ratio < best_ratio:
                best_ratio = ratio
                best_split = split

        splits[best_split].extend(items)
        for _, _, _, sc in items:
            split_scenario[best_split][sc] += 1

    return splits, split_scenario


def assign_individuals(items, scenario_totals, ratios):
    train_val_test = ("train", "val", "test_global")
    splits = {s: [] for s in train_val_test}
    split_scenario = {s: Counter() for s in train_val_test}
    target = {s: Counter() for s in train_val_test}
    for sc, total in scenario_totals.items():
        for ratio, split in zip(ratios, train_val_test):
            target[split][sc] = total * ratio

    for fname, src_img, lbl, sc in items:
        best_split = None
        best_ratio = float("inf")
        for split in train_val_test:
            t = target[split][sc]
            current = split_scenario[split][sc]
            if t > 0:
                ratio = current / t
            else:
                ratio = float("inf")
            if ratio < best_ratio:
                best_ratio = ratio
                best_split = split
        splits[best_split].append((fname, src_img, lbl))
        split_scenario[best_split][sc] += 1

    return splits


def simple_ratio_split(items, ratios):
    train_val_test = ("train", "val", "test_global")
    random.shuffle(items)
    n = len(items)
    train_end = int(n * ratios[0])
    val_end = train_end + int(n * ratios[1])
    splits = {s: [] for s in train_val_test}
    for i, item in enumerate(items):
        split = "train" if i < train_end else "val" if i < val_end else "test_global"
        splits[split].append(item)
    return splits


def write_yaml(path, test_split, base_path):
    with open(path, "w") as f:
        f.write(f"path: {base_path}\n")
        f.write("train: train/images\n")
        f.write("val: val/images\n")
        f.write(f"test: {test_split}/images\n")
        f.write("\n")
        f.write(f"nc: {len(CLASS_NAMES)}\n")
        f.write("names:\n")
        for name in CLASS_NAMES:
            f.write(f"  - {name}\n")
    print(f"Wrote {path}")


def main():
    random.seed(SEED)

    for split in SPLITS:
        for sub in ("images", "labels"):
            p = os.path.join(OUTPUT_DIR, split, sub)
            if os.path.isdir(p):
                shutil.rmtree(p)
            os.makedirs(p)

    seen_dst = set()
    class_counts = Counter()
    split_instances = {s: 0 for s in SPLITS}
    per_split = {s: Counter() for s in SPLITS}

    all_split_scenario = Counter()

    print("[sim] collecting...")
    sim_items = collect_images(SIM_DIRS)
    print(f"[sim] collected {len(sim_items)} images")

    sim_grouped = defaultdict(list)
    sim_ungrouped = []
    for fname, src_img, lbl in sim_items:
        mine_key, scenario = parse_sim_filename(fname)
        if mine_key and scenario:
            sim_grouped[(mine_key, scenario)].append((fname, src_img, lbl, scenario))
        else:
            sim_ungrouped.append((fname, src_img, lbl, scenario or "unknown"))

    sim_scenario_totals = Counter()
    for items in sim_grouped.values():
        for _, _, _, sc in items:
            sim_scenario_totals[sc] += 1
    for _, _, _, sc in sim_ungrouped:
        sim_scenario_totals[sc] += 1

    all_split_scenario += sim_scenario_totals

    sorted_groups = sorted(sim_grouped.items(), key=lambda x: -len(x[1]))
    sim_splits, sim_split_scenario = greedy_assign_groups(sorted_groups, sim_scenario_totals, SIM_SPLIT)

    if sim_ungrouped:
        sim_individual = assign_individuals(sim_ungrouped, sim_scenario_totals, SIM_SPLIT)
        for split in sim_splits:
            sim_splits[split].extend(sim_individual[split])

    print(f"\n[sim] scenario balance:")
    for sc in sorted(sim_scenario_totals):
        parts = []
        for s in ("train", "val", "test_global"):
            parts.append(f"{s}={sim_split_scenario[s].get(sc, 0)}")
        print(f"  {sc:20s} total={sim_scenario_totals[sc]:5d}  {' '.join(parts)}")

    for split in ("train", "val", "test_global"):
        for item in sim_splits[split]:
            fname, src_img, lbl = item[0], item[1], item[2]
            prefix = "sim"
            dst_name = f"{prefix}_{fname}"
            dst_stem = os.path.splitext(dst_name)[0]
            if dst_stem in seen_dst:
                continue
            seen_dst.add(dst_stem)
            _write_item(prefix, fname, src_img, lbl, split, dst_name, dst_stem,
                        SIM_ID_MAP, OUTPUT_DIR, per_split, class_counts, split_instances)

    print("\n[real1] collecting...")
    real1_items = collect_images([REAL1_DIR])
    print(f"[real1] collected {len(real1_items)} images")

    real1_with_scenario = [(f, s, l, "real1") for f, s, l in real1_items]
    real1_totals = Counter({"real1": len(real1_items)})
    all_split_scenario += real1_totals

    real1_splits = simple_ratio_split(real1_with_scenario, REAL_SPLIT)
    real1_split_scenario = Counter()
    for split, items in real1_splits.items():
        real1_split_scenario[split] = Counter({"real1": len(items)})

    print(f"[real1] split: train={len(real1_splits['train'])} val={len(real1_splits['val'])} test={len(real1_splits['test_global'])}")

    real1_test_pool = []
    for split in ("train", "val", "test_global"):
        for fname, src_img, lbl, _ in real1_splits[split]:
            prefix = "real1"
            dst_name = f"{prefix}_{fname}"
            dst_stem = os.path.splitext(dst_name)[0]
            if dst_stem in seen_dst:
                continue
            seen_dst.add(dst_stem)
            _write_item(prefix, fname, src_img, lbl, split, dst_name, dst_stem,
                        REAL1_ID_MAP, OUTPUT_DIR, per_split, class_counts, split_instances)
            if split == "test_global":
                real1_test_pool.append((dst_name, dst_stem))

    for dst_name, dst_stem in real1_test_pool:
        src_img = os.path.join(OUTPUT_DIR, "test_global", "images", dst_name)
        src_lbl = os.path.join(OUTPUT_DIR, "test_global", "labels", dst_stem + ".txt")
        if not os.path.exists(src_img):
            continue
        out_img = os.path.join(OUTPUT_DIR, "test_real_only", "images", dst_name)
        out_lbl = os.path.join(OUTPUT_DIR, "test_real_only", "labels", dst_stem + ".txt")
        if not os.path.exists(out_img):
            safe_link(src_img, out_img)
            shutil.copy2(src_lbl, out_lbl)
            per_split["test_real_only"]["images"] += 1
            if os.path.getsize(src_lbl) == 0:
                per_split["test_real_only"]["background"] += 1
            else:
                per_split["test_real_only"]["labeled"] += 1
                inst = sum(1 for _ in open(src_lbl))
                split_instances["test_real_only"] += inst

    print("\n[real2] collecting...")
    real2_items = collect_images([REAL2_DIR])
    print(f"[real2] collected {len(real2_items)} images")

    real2_with_scenario = [(f, s, l, "real2") for f, s, l in real2_items]
    real2_totals = Counter({"real2": len(real2_items)})
    all_split_scenario += real2_totals

    real2_splits = simple_ratio_split(real2_with_scenario, REAL_SPLIT)

    print(f"[real2] split: train={len(real2_splits['train'])} val={len(real2_splits['val'])} test={len(real2_splits['test_global'])}")

    real2_test_pool = []
    for split in ("train", "val", "test_global"):
        for fname, src_img, lbl, _ in real2_splits[split]:
            prefix = "real2"
            dst_name = f"{prefix}_{fname}"
            dst_stem = os.path.splitext(dst_name)[0]
            if dst_stem in seen_dst:
                continue
            seen_dst.add(dst_stem)
            _write_item(prefix, fname, src_img, lbl, split, dst_name, dst_stem,
                        REAL2_ID_MAP, OUTPUT_DIR, per_split, class_counts, split_instances)
            if split == "test_global":
                real2_test_pool.append((dst_name, dst_stem))

    for dst_name, dst_stem in real2_test_pool:
        src_img = os.path.join(OUTPUT_DIR, "test_global", "images", dst_name)
        src_lbl = os.path.join(OUTPUT_DIR, "test_global", "labels", dst_stem + ".txt")
        if not os.path.exists(src_img):
            continue
        out_img = os.path.join(OUTPUT_DIR, "test_real_only", "images", dst_name)
        out_lbl = os.path.join(OUTPUT_DIR, "test_real_only", "labels", dst_stem + ".txt")
        if not os.path.exists(out_img):
            safe_link(src_img, out_img)
            shutil.copy2(src_lbl, out_lbl)
            per_split["test_real_only"]["images"] += 1
            if os.path.getsize(src_lbl) == 0:
                per_split["test_real_only"]["background"] += 1
            else:
                per_split["test_real_only"]["labeled"] += 1
                inst = sum(1 for _ in open(src_lbl))
                split_instances["test_real_only"] += inst

    abs_out = os.path.abspath(OUTPUT_DIR)
    write_yaml(os.path.join(OUTPUT_DIR, "data_global.yaml"), "test_global", abs_out)
    write_yaml(os.path.join(OUTPUT_DIR, "data_real_only.yaml"), "test_real_only", abs_out)

    print("\n=== Dataset Summary ===")
    for split in SPLITS:
        c = per_split[split]
        imgs = c["images"]
        labeled = c.get("labeled", 0)
        bg = c.get("background", 0)
        print(f"  {split:15s}: {imgs:6d} images ({labeled} labeled, {bg} background, {split_instances[split]} instances)")

    print("\n=== Per-class instances ===")
    for i, name in enumerate(CLASS_NAMES):
        print(f"  {i}: {name}: {class_counts.get(i, 0)}")

    total = sum(per_split[s]["images"] for s in SPLITS)
    print(f"\nTotal images (incl. duplicates in test_real_only): {total}")
    print(f"Done! Output at: {OUTPUT_DIR}")


def _write_item(prefix, fname, src_img, lbl, split, dst_name, dst_stem,
                id_map, output_dir, per_split, class_counts, split_instances):
    out_img = os.path.join(output_dir, split, "images", dst_name)
    out_lbl = os.path.join(output_dir, split, "labels", dst_stem + ".txt")

    safe_link(src_img, out_img)
    per_split[split]["images"] += 1

    if lbl and os.path.exists(lbl):
        if os.path.getsize(lbl) == 0:
            with open(out_lbl, "w"):
                pass
            per_split[split]["background"] += 1
        else:
            convert_label(lbl, out_lbl, id_map)
            inst = 0
            with open(out_lbl, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if parts:
                        inst += 1
                        class_counts[int(parts[0])] += 1
            split_instances[split] += inst
            per_split[split]["labeled"] += 1
    else:
        with open(out_lbl, "w"):
            pass
        per_split[split]["background"] += 1


if __name__ == "__main__":
    main()
