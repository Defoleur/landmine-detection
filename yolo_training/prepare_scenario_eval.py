import json
import os
import re
import shutil
from collections import Counter, defaultdict

from ultralytics import YOLO

SEED = 42

SOURCE_DIR = "/media/geekscode/Developer1/Projects/gd-autopilot/datasets/dataset_auto_ultimate_test_20260824_203515"
MODEL_PATH = "/media/geekscode/Developer1/Projects/gd-autopilot/yolo_training/trained_models_project/landmine_multimodal_oblique_7cls_e150_grouped/weights/best.pt"
OUTPUT_DIR = "/media/geekscode/Developer1/Projects/gd-autopilot/yolo_training/datasets/dataset_scenario_eval_ultimate_summer_model"
DEVICE = 0

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

SCENARIO_RE = re.compile(r"_(spring|summer|autumn|winter)_(day|morning|evening)\.png$", re.IGNORECASE)

EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")

SEASONS = ["spring", "summer", "autumn", "winter"]
TIMES = ["day", "morning", "evening"]
SCENARIOS = [f"{s}_{t}" for s in SEASONS for t in TIMES]

def quad_to_hbb(coords):
    xs = coords[0::2]
    ys = coords[1::2]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    return (x_min + x_max) / 2.0, (y_min + y_max) / 2.0, x_max - x_min, y_max - y_min


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


def write_yaml(path, base_path):
    with open(path, "w") as f:
        f.write(f"path: {base_path}\n")
        f.write("train: images\n")
        f.write("val: images\n")
        f.write("\n")
        f.write(f"nc: {len(CLASS_NAMES)}\n")
        f.write("names:\n")
        for name in CLASS_NAMES:
            f.write(f"  - {name}\n")


def split_by_scenario():
    src_img_dir = os.path.join(SOURCE_DIR, "images")
    src_lbl_dir = os.path.join(SOURCE_DIR, "labels")

    scenario_items = defaultdict(list)
    unmatched = 0

    for fname in sorted(os.listdir(src_img_dir)):
        if not fname.lower().endswith(EXTENSIONS):
            continue
        m = SCENARIO_RE.search(fname)
        if not m:
            unmatched += 1
            continue
        scenario = f"{m.group(1)}_{m.group(2)}"
        stem = os.path.splitext(fname)[0]
        lbl = os.path.join(src_lbl_dir, stem + ".txt")
        scenario_items[scenario].append((fname, lbl if os.path.exists(lbl) else None))

    if unmatched:
        print(f"  WARN: {unmatched} images without scenario match")

    print(f"\n{'Scenario':<20} {'Images':>7}")
    print("-" * 28)
    for sc in SCENARIOS:
        n = len(scenario_items.get(sc, []))
        print(f"  {sc:<20} {n:>7d}")
    print(f"  {'TOTAL':<20} {sum(len(v) for v in scenario_items.values()):>7d}")

    return scenario_items


def prepare_scenario_folders(scenario_items):
    src_img_dir = os.path.join(SOURCE_DIR, "images")
    prepared = {}

    for scenario in SCENARIOS:
        items = scenario_items.get(scenario, [])
        if not items:
            print(f"  SKIP: {scenario} (empty)")
            continue

        sc_dir = os.path.join(OUTPUT_DIR, scenario)
        if os.path.isdir(sc_dir):
            shutil.rmtree(sc_dir)
        os.makedirs(os.path.join(sc_dir, "images"))
        os.makedirs(os.path.join(sc_dir, "labels"))

        class_counts = Counter()
        n_labeled = 0
        n_bg = 0

        for fname, lbl in items:
            src_img = os.path.join(src_img_dir, fname)
            stem = os.path.splitext(fname)[0]
            out_img = os.path.join(sc_dir, "images", fname)
            out_lbl = os.path.join(sc_dir, "labels", stem + ".txt")

            safe_link(src_img, out_img)

            if lbl and os.path.exists(lbl) and os.path.getsize(lbl) > 0:
                convert_label(lbl, out_lbl)
                n_labeled += 1
                with open(out_lbl) as f:
                    for line in f:
                        p = line.strip().split()
                        if p:
                            class_counts[int(p[0])] += 1
            else:
                with open(out_lbl, "w"):
                    pass
                n_bg += 1

        write_yaml(os.path.join(sc_dir, "data.yaml"), os.path.abspath(sc_dir))

        prepared[scenario] = {
            "dir": sc_dir,
            "total": len(items),
            "labeled": n_labeled,
            "bg": n_bg,
            "class_counts": class_counts,
        }

        print(f"  {scenario:<20} {len(items):>5d} images  ({n_labeled} labeled, {n_bg} bg)")

    return prepared


def evaluate_per_scenario(prepared):
    model = YOLO(MODEL_PATH)
    metrics = {}

    for scenario in SCENARIOS:
        if scenario not in prepared:
            continue
        sc_dir = prepared[scenario]["dir"]
        yaml_path = os.path.join(sc_dir, "data.yaml")

        print(f"\n  Evaluating {scenario}...")
        results = model.val(data=yaml_path, split="val", device=DEVICE, verbose=False)

        class_ap = {}
        for i, idx in enumerate(results.ap_class_index):
            class_ap[CLASS_NAMES[idx]] = float(results.maps[i])

        metrics[scenario] = {
            "mAP50": float(results.box.map50),
            "mAP50-95": float(results.box.map),
            "precision": float(results.box.mp),
            "recall": float(results.box.mr),
            "per_class_ap": class_ap,
            "images": prepared[scenario]["total"],
            "labeled": prepared[scenario]["labeled"],
        }

    return metrics


def main():
    if os.path.isdir(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR)

    print("=" * 60)
    print("  SCENARIO EVALUATION PIPELINE")
    print("=" * 60)

    print("\n[1/4] Splitting dataset by scenario...")
    scenario_items = split_by_scenario()

    print("\n[2/4] Preparing scenario folders (hardlink + label conversion)...")
    prepared = prepare_scenario_folders(scenario_items)

    print("\n[3/4] Evaluating model per scenario...")
    metrics = evaluate_per_scenario(prepared)

    metrics_path = os.path.join(OUTPUT_DIR, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\n  Metrics saved to {metrics_path}")

    print(f"\n{'Scenario':<20} {'Images':>6} {'mAP50':>7} {'mAP50-95':>9} {'Prec':>7} {'Rec':>7}")
    print("-" * 58)
    for sc in SCENARIOS:
        if sc not in metrics:
            continue
        m = metrics[sc]
        print(f"  {sc:<18} {m['images']:>6d} {m['mAP50']:>7.4f} {m['mAP50-95']:>9.4f} {m['precision']:>7.4f} {m['recall']:>7.4f}")

    print(f"\n[4/4] Done! Run 'python yolo_training/plot_scenario_metrics.py' to generate plots.")


if __name__ == "__main__":
    main()
