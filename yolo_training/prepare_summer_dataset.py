import os
import random
import shutil
from collections import Counter

SEED = 42
SEASON_FILTER = "summer_day"

SIM_DIRS = [
    "/media/geekscode/Developer1/Projects/gd-autopilot/datasets/dataset_auto_last_20260814_180837",
    "/media/geekscode/Developer1/Projects/gd-autopilot/datasets/dataset_auto_last_20260814_180837_empty",
]
OUTPUT_DIR = "/media/geekscode/Developer1/Projects/gd-autopilot/yolo_training/datasets/dataset_summer_day"

CLASS_NAMES = [
    "mine_disc",
    "mine_rectangular_elongated",
    "mine_square",
    "mine_rectangular",
    "mine_cylinder_vert",
    "mine_directional_standing",
    "mine_round_directional_standing",
]

SIM_ID_MAP = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 6: 5, 7: 6}

SPLIT_RATIOS = (0.80, 0.10, 0.10)

EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")
SPLITS = ("train", "val", "test")


def quad_to_hbb(coords):
    xs = coords[0::2]
    ys = coords[1::2]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    x_c = (x_min + x_max) / 2.0
    y_c = (y_min + y_max) / 2.0
    w = x_max - x_min
    h = y_max - y_min
    return x_c, y_c, w, h


def convert_label(src_path, dst_path):
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
            new_id = SIM_ID_MAP.get(old_id)
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
            x_c = max(0.0, min(1.0, x_c))
            y_c = max(0.0, min(1.0, y_c))
            w = max(0.0, min(1.0, w))
            h = max(0.0, min(1.0, h))
            lines_out.append(f"{new_id} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}")

    with open(dst_path, "w") as f:
        if lines_out:
            f.write("\n".join(lines_out) + "\n")


def safe_link(src, dst):
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def collect_summer_images(src_dirs):
    collected = []
    for src in src_dirs:
        if not os.path.isdir(src):
            print(f"  WARN: source not found: {src}")
            continue
        images_dir = os.path.join(src, "images")
        labels_dir = os.path.join(src, "labels")
        if not os.path.isdir(images_dir):
            print(f"  WARN: no images dir: {images_dir}")
            continue
        for fname in sorted(os.listdir(images_dir)):
            if not fname.lower().endswith(EXTENSIONS):
                continue
            if not fname.endswith(f"_{SEASON_FILTER}.png") and not fname.lower().endswith(f"_{SEASON_FILTER}.jpg"):
                continue
            stem = os.path.splitext(fname)[0]
            lbl = os.path.join(labels_dir, stem + ".txt")
            collected.append((fname, lbl if os.path.exists(lbl) else None))
    return collected


def write_yaml(path, base_path):
    with open(path, "w") as f:
        f.write(f"path: {base_path}\n")
        f.write("train: train/images\n")
        f.write("val: val/images\n")
        f.write("test: test/images\n")
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
            os.makedirs(os.path.join(OUTPUT_DIR, split, sub), exist_ok=True)

    items = collect_summer_images(SIM_DIRS)
    print(f"Collected {len(items)} summer_day images")

    random.shuffle(items)
    n = len(items)
    train_end = int(n * SPLIT_RATIOS[0])
    val_end = train_end + int(n * SPLIT_RATIOS[1])

    per_split = {s: Counter() for s in SPLITS}
    class_counts = Counter()
    split_instances = {s: 0 for s in SPLITS}
    seen_dst = set()

    for i, (fname, lbl) in enumerate(items):
        if i < train_end:
            split = "train"
        elif i < val_end:
            split = "val"
        else:
            split = "test"

        stem = os.path.splitext(fname)[0]
        if stem in seen_dst:
            print(f"  WARN: collision, skipping {fname}")
            continue
        seen_dst.add(stem)

        out_img = os.path.join(OUTPUT_DIR, split, "images", fname)
        out_lbl = os.path.join(OUTPUT_DIR, split, "labels", stem + ".txt")

        src_img = None
        for d in SIM_DIRS:
            cand = os.path.join(d, "images", fname)
            if os.path.exists(cand):
                src_img = cand
                break
        if src_img is None:
            print(f"  WARN: image not found for {fname}")
            continue

        safe_link(src_img, out_img)
        per_split[split]["images"] += 1

        if lbl and os.path.exists(lbl):
            if os.path.getsize(lbl) == 0:
                with open(out_lbl, "w"):
                    pass
                per_split[split]["background"] += 1
            else:
                convert_label(lbl, out_lbl)
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

    write_yaml(os.path.join(OUTPUT_DIR, "data.yaml"), os.path.abspath(OUTPUT_DIR))

    print("\n=== Summer Dataset Summary ===")
    for split in SPLITS:
        c = per_split[split]
        imgs = c["images"]
        labeled = c.get("labeled", 0)
        bg = c.get("background", 0)
        print(f"  {split:10s}: {imgs:5d} images ({labeled} labeled, {bg} background, {split_instances[split]} instances)")

    print("\n=== Per-class instances (target taxonomy) ===")
    for i, name in enumerate(CLASS_NAMES):
        print(f"  {i}: {name}: {class_counts.get(i, 0)}")

    print(f"\nTotal images: {sum(per_split[s]['images'] for s in SPLITS)}")
    print(f"Done! Output at: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()