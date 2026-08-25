import cv2
import os
import glob
import re
import numpy as np
import random

DATASET_DIR = "/media/geekscode/Developer1/Projects/gd-autopilot/datasets/dataset_auto_ultimate_test_20260824_203515"

CLASS_NAMES = {
    0: "mine_disc",
    1: "mine_rectangular_elongated",
    2: "mine_square",
    3: "mine_rectangular",
    4: "mine_cylinder_vert",
    5: "mine_butterfly",
    6: "mine_directional_standing",
    7: "mine_round_directional_standing",
}

EXACT_CLASS_NAMES = {
    0: "Mine_Cloth",
    1: "Mine_Dzika",
    2: "Mine_ML7",
    3: "Mine_ML8",
    4: "Mine_MOB",
    5: "Mine_MON100",
    6: "Mine_MON200",
    7: "Mine_MON50",
    8: "Mine_MON90",
    9: "Mine_MS3",
    10: "Mine_MS4",
    11: "Mine_OZM72",
    12: "Mine_PFM1",
    13: "Mine_PMD6",
    14: "Mine_PMN",
    15: "Mine_PMN2",
    16: "Mine_PMN4",
    17: "Mine_POB_Pilka",
    18: "Mine_POM2",
    19: "Mine_POM3",
    20: "Mine_POMZ2",
    21: "Mine_PTKM1R",
    22: "Mine_PTM1",
    23: "Mine_PTM3",
    24: "Mine_PTM4",
    25: "Mine_TM46",
    26: "Mine_TM57",
    27: "Mine_TM62",
    28: "Mine_TM62B",
    29: "Mine_TM62D",
    30: "Mine_TM62P",
    31: "Mine_TM62P2",
    32: "Mine_TM62P3",
    33: "Mine_TM62T",
    34: "Mine_TM72",
    35: "Mine_TM83",
    36: "Mine_TM89",
}

COLORS = {}

def get_color(cls_id):
    if cls_id not in COLORS:
        COLORS[cls_id] = (random.randint(50, 255), random.randint(50, 255), random.randint(50, 255))
    return COLORS[cls_id]

def load_class_names(classes_path):
    names = {}
    if classes_path and os.path.exists(classes_path):
        with open(classes_path, 'r') as f:
            for i, line in enumerate(f):
                name = line.strip()
                if name:
                    names[i] = name
    return names

def parse_filename(basename):
    parts = os.path.splitext(basename)[0].split("_")
    ci = next((i for i in range(len(parts) - 1, -1, -1) if re.fullmatch(r"\d{6}", parts[i])), None)
    if ci is None:
        return {"counter": "", "mine": "", "season": "", "daytime": ""}
    if ci == len(parts) - 1:
        return {"counter": parts[ci], "mine": "", "season": "", "daytime": ""}
    return {
        "counter": parts[ci],
        "mine": "_".join(parts[ci + 1:-2]),
        "season": parts[-2],
        "daytime": parts[-1],
    }

def draw_labels_on_image(img, label_path, class_names=None):
    h_img, w_img = img.shape[:2]

    if not os.path.exists(label_path):
        cv2.putText(img, "No Labels", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        return img

    with open(label_path, 'r') as f:
        lines = f.readlines()

    for line in lines:
        parts = list(map(float, line.strip().split()))

        if len(parts) == 0: continue

        cls_id = int(parts[0])
        coords = parts[1:]
        color = get_color(cls_id)

        label_text = (class_names or {}).get(cls_id) or CLASS_NAMES.get(cls_id, f"ID: {cls_id}")

        if len(coords) == 4:
            nx, ny, nw, nh = coords

            w_box = int(nw * w_img)
            h_box = int(nh * h_img)
            x_center = int(nx * w_img)
            y_center = int(ny * h_img)
            
            x1 = int(x_center - w_box / 2)
            y1 = int(y_center - h_box / 2)
            x2 = x1 + w_box
            y2 = y1 + h_box

            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            
            (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(img, (x1, y1 - 20), (x1 + tw, y1), color, -1)
            cv2.putText(img, label_text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)

        elif len(coords) >= 8:
            points = []
            for i in range(0, len(coords), 2):
                px = int(coords[i] * w_img)
                py = int(coords[i+1] * h_img)
                points.append([px, py])
            
            pts_np = np.array(points, np.int32).reshape((-1, 1, 2))
            
            cv2.polylines(img, [pts_np], isClosed=True, color=color, thickness=2)
            
            start_x, start_y = points[0]
            (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(img, (start_x, start_y - 20), (start_x + tw, start_y), color, -1)
            cv2.putText(img, label_text, (start_x, start_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)

    return img

def main():
    images_dir = os.path.join(DATASET_DIR, "images")
    labels_dir = os.path.join(DATASET_DIR, "labels")
    exact_dir = os.path.join(DATASET_DIR, "exact_labels")

    deleted_dir = DATASET_DIR + "_deleted"
    deleted_images_dir = os.path.join(deleted_dir, "images")
    deleted_labels_dir = os.path.join(deleted_dir, "labels")
    deleted_exact_dir = os.path.join(deleted_dir, "exact_labels")

    exact_class_names = load_class_names(os.path.join(exact_dir, "classes.txt"))
    merged_class_names = load_class_names(os.path.join(labels_dir, "classes.txt"))

    extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']

    def gather_entries(base_images_dir, deleted):
        entries = []
        for ext in extensions:
            for path in glob.glob(os.path.join(base_images_dir, ext)):
                entries.append({
                    "img_path": path,
                    "deleted": deleted,
                })
        return entries

    entries = gather_entries(images_dir, False)
    if os.path.isdir(deleted_images_dir):
        entries += gather_entries(deleted_images_dir, True)
    entries.sort(key=lambda e: os.path.basename(e["img_path"]))

    if not entries:
        print(f"Error: No images found in {images_dir}")
        print("Check your DATASET_DIR path.")
        return

    def label_paths(entry):
        name_only = os.path.splitext(os.path.basename(entry["img_path"]))[0]
        if entry["deleted"]:
            return {
                "images": os.path.join(deleted_images_dir, name_only + os.path.splitext(entry["img_path"])[1]),
                "labels": os.path.join(deleted_labels_dir, name_only + ".txt"),
                "exact": os.path.join(deleted_exact_dir, name_only + ".txt"),
            }
        return {
            "images": os.path.join(images_dir, name_only + os.path.splitext(entry["img_path"])[1]),
            "labels": os.path.join(labels_dir, name_only + ".txt"),
            "exact": os.path.join(exact_dir, name_only + ".txt"),
        }

    def move_entry(entry, to_deleted):
        paths = label_paths(entry)
        src_map = {
            "images": entry["img_path"],
            "labels": paths["labels"],
            "exact": paths["exact"],
        }
        if to_deleted:
            dst_map = {
                "images": os.path.join(deleted_images_dir, os.path.basename(entry["img_path"])),
                "labels": os.path.join(deleted_labels_dir, os.path.basename(paths["labels"])),
                "exact": os.path.join(deleted_exact_dir, os.path.basename(paths["exact"])),
            }
        else:
            dst_map = {
                "images": os.path.join(images_dir, os.path.basename(entry["img_path"])),
                "labels": os.path.join(labels_dir, os.path.basename(paths["labels"])),
                "exact": os.path.join(exact_dir, os.path.basename(paths["exact"])),
            }

        for folder in (deleted_images_dir, deleted_labels_dir, deleted_exact_dir,
                       images_dir, labels_dir, exact_dir):
            os.makedirs(folder, exist_ok=True)

        for key in ("images", "labels", "exact"):
            src = src_map[key]
            dst = dst_map[key]
            if os.path.exists(src):
                os.replace(src, dst)

        entry["deleted"] = to_deleted
        entry["img_path"] = dst_map["images"]
        return entry

    print(f"Loaded {len(entries)} images.")
    print("Controls:")
    print(" [D] / [Right Arrow] : Next Image")
    print(" [A] / [Left Arrow]  : Previous Image")
    print(" ]                    : Next category (mine+season+daytime)")
    print(" [                    : Previous category")
    print(" [G]                 : Go to image by number")
    print(" [F]                 : Filter by class (empty input to clear)")
    print(" [E]                 : Toggle merged / exact labels")
    print(" [X]                 : Delete current image")
    print(" [C]                 : Delete category (mine+season+daytime)")
    print(" [R]                 : Restore current image")
    print(" [B]                 : Restore category")
    print(" [T]                 : Toggle show deleted entries")
    print(" [Q] / [Esc]         : Quit")

    idx = 0
    use_exact = False
    show_deleted = True
    class_filter = None

    def get_active_names():
        return (exact_class_names or EXACT_CLASS_NAMES) if use_exact else (merged_class_names or CLASS_NAMES)

    def entry_matches_filter(entry):
        if class_filter is None:
            return True
        paths = label_paths(entry)
        active_dir = paths["exact"] if use_exact else paths["labels"]
        if not os.path.exists(active_dir):
            return False
        with open(active_dir, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue
                try:
                    if int(parts[0]) == class_filter:
                        return True
                except ValueError:
                    continue
        return False

    def entry_category(entry):
        meta = parse_filename(os.path.basename(entry["img_path"]))
        return (meta["mine"], meta["season"], meta["daytime"])

    def describe_category(cat):
        return " | ".join(p for p in cat if p) or "manual"

    def visible_entries():
        base = entries if show_deleted else [e for e in entries if not e["deleted"]]
        if class_filter is None:
            return base
        return [e for e in base if entry_matches_filter(e)]

    while True:
        vis = visible_entries()
        if not vis:
            print("No entries to show. Press T to show deleted.")
            show_deleted = True
            vis = entries
            if not vis:
                break
        if idx >= len(vis):
            idx = 0

        entry = vis[idx]

        img = cv2.imread(entry["img_path"])

        if img is None:
            print(f"Failed to load: {entry['img_path']}")
            idx = (idx + 1) % len(vis)
            continue

        paths = label_paths(entry)
        active_dir = paths["exact"] if use_exact else paths["labels"]

        active_names = (exact_class_names or EXACT_CLASS_NAMES) if use_exact else None
        display_img = draw_labels_on_image(img.copy(), active_dir, active_names)

        basename = os.path.basename(entry["img_path"])
        meta = parse_filename(basename)
        mode_tag = "EXACT" if use_exact else "MERGED"
        filter_tag = ""
        if class_filter is not None:
            filter_name = get_active_names().get(class_filter, f"ID {class_filter}")
            filter_tag = f" [FILTER: {filter_name}]"
        info = f"({idx+1}/{len(vis)}) [{mode_tag}]{filter_tag}"
        cv2.putText(display_img, info, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 4)
        cv2.putText(display_img, info, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

        mine = meta["mine"]
        if mine.startswith("obstacle_mine_"):
            mine = mine[len("obstacle_mine_"):]
        ctx = " | ".join(p for p in (
            f"Img: {meta['counter']}",
            f"Mine: {mine}" if mine else "",
            meta["season"],
            meta["daytime"],
        ) if p)
        if ctx:
            cv2.putText(display_img, ctx, (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 4)
            cv2.putText(display_img, ctx, (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        if entry["deleted"]:
            cv2.rectangle(display_img, (0, 0), (display_img.shape[1] - 1, display_img.shape[0] - 1), (0, 0, 255), 3)
            cv2.putText(display_img, "DELETED", (display_img.shape[1] - 220, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)

        cv2.imshow("Dataset Viewer", display_img)

        key = cv2.waitKey(0)

        if key in [ord('d'), 83, 32]:
            idx = (idx + 1) % len(vis)
        elif key in [ord('a'), 81]:
            idx = (idx - 1) % len(vis)
        elif key in [93]:
            cat = entry_category(entry)
            for i in range(1, len(vis)):
                j = (idx + i) % len(vis)
                if entry_category(vis[j]) != cat:
                    idx = j
                    break
        elif key in [91]:
            cat = entry_category(entry)
            for i in range(1, len(vis)):
                j = (idx - i) % len(vis)
                if entry_category(vis[j]) != cat:
                    idx = j
                    break
        elif key in [ord('g'), ord('G')]:
            try:
                num = int(input(f"Enter image number (1-{len(vis)}): "))
                if 1 <= num <= len(vis):
                    idx = num - 1
                else:
                    print(f"Number out of range. Valid range: 1-{len(vis)}")
            except ValueError:
                print("Invalid number. Please enter an integer.")
        elif key in [ord('e'), 69]:
            use_exact = not use_exact
            print(f"Switched to {'exact' if use_exact else 'merged'} labels.")
        elif key in [ord('f'), ord('F')]:
            names = get_active_names()
            print(f"Classes in current mode ({len(names)}):")
            print(", ".join(f"{i}:{name}" for i, name in sorted(names.items())))
            raw = input("Enter class ID or name (empty to clear filter): ").strip()
            if not raw:
                class_filter = None
                idx = 0
                print("Filter cleared.")
            else:
                new_filter = None
                if raw.isdigit():
                    new_filter = int(raw)
                else:
                    raw_lower = raw.lower()
                    for cid, name in names.items():
                        if name.lower() == raw_lower or raw_lower in name.lower():
                            new_filter = cid
                            break
                if new_filter is None:
                    print(f"Class '{raw}' not found in current mode.")
                else:
                    class_filter = new_filter
                    idx = 0
                    matched = sum(1 for e in visible_entries() if entry_matches_filter(e))
                    print(f"Filtering by class {names.get(class_filter, class_filter)}: {matched} images.")
        elif key in [ord('r'), 82]:
            if entry["deleted"]:
                move_entry(entry, False)
                print(f"Restored: {basename}")
            else:
                print(f"Entry is not deleted: {basename}")
        elif key in [ord('b'), 66]:
            if entry["deleted"]:
                cat = entry_category(entry)
                targets = [e for e in entries if e["deleted"] and entry_category(e) == cat]
                for e in targets:
                    move_entry(e, False)
                print(f"Restored {len(targets)} images of category '{describe_category(cat)}'.")
            else:
                print(f"Entry is not deleted: {basename}")
        elif key in [ord('t'), 84]:
            current = entry["img_path"]
            show_deleted = not show_deleted
            vis = visible_entries()
            if not vis:
                show_deleted = not show_deleted
                vis = visible_entries()
            found = [i for i, e in enumerate(vis) if e["img_path"] == current]
            if found:
                idx = found[0]
            elif show_deleted:
                idx = 0
            else:
                cur_pos = next((i for i, e in enumerate(entries) if e["img_path"] == current), -1)
                next_visible = [i for i, e in enumerate(entries)
                                if i > cur_pos and not e["deleted"]]
                if next_visible:
                    idx = next(vis_i for vis_i, e in enumerate(vis)
                               if e["img_path"] == entries[next_visible[0]]["img_path"])
                else:
                    idx = 0
            print(f"Show deleted entries: {'ON' if show_deleted else 'OFF'}")
        elif key in [ord('x'), ord('X')]:
            if not entry["deleted"]:
                move_entry(entry, True)
                print(f"Moved to deleted: {basename}")
            else:
                print(f"Entry already deleted: {basename}")
        elif key in [ord('c'), ord('C')]:
            if entry["deleted"]:
                print(f"Entry already deleted: {basename}")
            else:
                cat = entry_category(entry)
                targets = [e for e in entries if not e["deleted"] and entry_category(e) == cat]
                cat_desc = describe_category(cat)
                if len(targets) == 1:
                    move_entry(entry, True)
                    print(f"Moved to deleted: {basename}")
                else:
                    for e in targets:
                        move_entry(e, True)
                    print(f"Deleted {len(targets)} images of category '{cat_desc}'.")
        elif key in [ord('q'), 27]:
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()