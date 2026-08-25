import cv2
import os
import glob
import numpy as np
import random

DATASET_DIR = "/media/geekscode/Developer1/Projects/gd-autopilot/yolo_training/datasets/lwir_real_all"

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
    0: "Mine_Cloth", 1: "Mine_Dzika", 2: "Mine_ML7", 3: "Mine_ML8",
    4: "Mine_MOB", 5: "Mine_MON100", 6: "Mine_MON200", 7: "Mine_MON50",
    8: "Mine_MON90", 9: "Mine_MS3", 10: "Mine_MS4", 11: "Mine_OZM72",
    12: "Mine_PFM1", 13: "Mine_PMD6", 14: "Mine_PMN", 15: "Mine_PMN2",
    16: "Mine_PMN4", 17: "Mine_POB_Pilka", 18: "Mine_POM2", 19: "Mine_POM3",
    20: "Mine_POMZ2", 21: "Mine_PTKM1R", 22: "Mine_PTM1", 23: "Mine_PTM3",
    24: "Mine_PTM4", 25: "Mine_TM46", 26: "Mine_TM57", 27: "Mine_TM62",
    28: "Mine_TM62B", 29: "Mine_TM62D", 30: "Mine_TM62P", 31: "Mine_TM62P2",
    32: "Mine_TM62P3", 33: "Mine_TM62T", 34: "Mine_TM72", 35: "Mine_TM83",
    36: "Mine_TM89",
}

WIN_NAME = "Label Editor"

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

def build_boxes(label_path):
    boxes = []
    if not os.path.exists(label_path):
        return boxes
    with open(label_path, 'r') as f:
        lines = f.readlines()
    for i, line in enumerate(lines):
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        try:
            floats = list(map(float, parts))
        except ValueError:
            continue
        boxes.append({
            "line_index": i,
            "cls_id": int(floats[0]),
            "coords": floats[1:],
            "poly": None,
        })
    return boxes

def to_pixels(box, w, h):
    coords = box["coords"]
    if len(coords) == 4:
        nx, ny, nw, nh = coords
        x1 = int((nx - nw / 2) * w)
        y1 = int((ny - nh / 2) * h)
        x2 = int((nx + nw / 2) * w)
        y2 = int((ny + nh / 2) * h)
        return np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], np.int32)
    pts = []
    for i in range(0, len(coords), 2):
        pts.append([int(coords[i] * w), int(coords[i + 1] * h)])
    return np.array(pts, np.int32)

def contains(poly, x, y):
    if poly is None or len(poly) < 3:
        return False
    return cv2.pointPolygonTest(poly, (float(x), float(y)), False) >= 0

def draw_box(img, box, w, h, names, show_text, selected=False):
    poly = box["poly"]
    color = get_color(box["cls_id"])

    if selected:
        cv2.polylines(img, [poly], isClosed=True, color=(0, 255, 255), thickness=4)
        overlay = img.copy()
        cv2.fillPoly(overlay, [poly], (0, 255, 255))
        cv2.addWeighted(overlay, 0.25, img, 0.75, 0, dst=img)
        cv2.polylines(img, [poly], isClosed=True, color=(0, 255, 255), thickness=4)
        for p in poly:
            cv2.circle(img, (int(p[0]), int(p[1])), 4, (0, 255, 255), -1)
    else:
        cv2.polylines(img, [poly], isClosed=True, color=color, thickness=2)

    if show_text:
        xs = [int(p[0]) for p in poly]
        ys = [int(p[1]) for p in poly]
        x1, y1 = min(xs), min(ys)
        label_text = (names or {}).get(box["cls_id"]) or CLASS_NAMES.get(box["cls_id"], f"ID: {box['cls_id']}")
        if selected:
            label_text = ">> SELECTED: " + label_text
        text_color = (0, 255, 255) if selected else color
        (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        txt_y = max(y1 - 6, th + 4)
        cv2.rectangle(img, (x1, txt_y - th - 2), (x1 + tw, txt_y + 2), text_color, -1)
        cv2.putText(img, label_text, (x1, txt_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

def rewrite_label_file(label_path, box, new_id):
    with open(label_path, 'r') as f:
        lines = f.readlines()
    parts = lines[box["line_index"]].rstrip("\n").split(" ")
    parts[0] = str(new_id)
    lines[box["line_index"]] = " ".join(parts) + "\n"
    tmp = label_path + ".tmp"
    with open(tmp, 'w') as f:
        f.writelines(lines)
    os.replace(tmp, label_path)
    old_id = box["cls_id"]
    box["cls_id"] = new_id
    return old_id

def hotkey_map(names):
    chars = list("0123456789") + [chr(c) for c in range(ord('a'), ord('z') + 1)]
    mapping = {}
    for i, cls_id in enumerate(sorted(names.keys())):
        if i < len(chars):
            mapping[chars[i]] = cls_id
    return mapping

def pick_class(img, names, current_cls):
    h, w = img.shape[:2]
    overlay = img.copy()
    pad = 10
    line_h = 26
    panel_w = 420
    entries = sorted(names.items())
    panel_h = pad * 2 + line_h * (len(entries) + 1)
    panel_h = min(panel_h, h - 20)

    xs, ys, xb, yb = w - panel_w - 10, 10, w - 10, panel_h + 10
    sub = overlay[ys:yb, xs:xb]
    mask = np.zeros(sub.shape[:2], np.uint8)
    cv2.rectangle(mask, (0, 0), (sub.shape[1] - 1, sub.shape[0] - 1), 255, -1)
    overlay[ys:yb, xs:xb] = cv2.bitwise_and(overlay[ys:yb, xs:xb], overlay[ys:yb, xs:xb], mask=mask)
    cv2.addWeighted(overlay[ys:yb, xs:xb], 0.45, np.full(sub.shape, (0, 0, 0), np.uint8), 0.55, 0,
                    dst=overlay[ys:yb, xs:xb])

    mapping = hotkey_map(names)
    y = ys + pad + line_h
    cv2.putText(overlay, f"Select class (current: {current_cls})", (xs + pad, y - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    for cls_id, name in entries:
        hk = next((k for k, v in mapping.items() if v == cls_id), '?')
        text = f"{hk}: [{cls_id}] {name}"
        color = get_color(cls_id)
        cv2.rectangle(overlay, (xs + pad, y - line_h + 6), (xs + panel_w - pad, y - 2), color, -1)
        cv2.putText(overlay, text, (xs + pad + 4, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)
        y += line_h

    cv2.putText(overlay, "ESC to cancel", (xs + pad, y + line_h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    cv2.imshow(WIN_NAME, overlay)
    while True:
        key = cv2.waitKey(0)
        if key in [27, ord('q')]:
            return None
        ch = chr(key) if key < 256 else ''
        if ch in mapping:
            return mapping[ch]

def main():
    images_dir = os.path.join(DATASET_DIR, "images")
    labels_dir = os.path.join(DATASET_DIR, "labels")
    exact_dir = os.path.join(DATASET_DIR, "exact_labels")

    exact_class_names = load_class_names(os.path.join(exact_dir, "classes.txt"))
    merged_class_names = load_class_names(os.path.join(labels_dir, "classes.txt"))

    extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    img_files = []
    for ext in extensions:
        img_files.extend(glob.glob(os.path.join(images_dir, ext)))
    img_files = sorted(img_files)

    if not img_files:
        print(f"Error: No images found in {images_dir}")
        return

    print(f"Loaded {len(img_files)} images.")
    print("Controls:")
    print(" [D] / [Right Arrow] : Next Image")
    print(" [A] / [Left Arrow]  : Previous Image")
    print(" [E]                 : Toggle merged / exact labels")
    print(" [L]                 : Toggle label text")
    print(" [Left Click]        : Select box (click again to deselect)")
    print(" [N]                 : Cycle overlapping boxes under cursor")
    print(" [M]                 : Reassign selected box class")
    print(" [Q] / [Esc]         : Quit")

    idx = 0
    use_exact = False
    show_text = True
    selected = None
    last_click = None
    boxes = []
    img_path = None
    label_path = None
    names = None

    def reload():
        nonlocal boxes, names
        active = exact_dir if use_exact else labels_dir
        n = os.path.basename(os.path.splitext(img_path)[0])
        label = os.path.join(active, n + ".txt")
        boxes = build_boxes(label)
        if use_exact:
            names = exact_class_names or EXACT_CLASS_NAMES
        else:
            names = merged_class_names or CLASS_NAMES
        return label

    def boxes_under(x, y):
        found = []
        for b in boxes:
            if contains(b["poly"], x, y):
                found.append(b)
        return found

    def on_mouse(event, mx, my, flags, param):
        nonlocal selected, last_click
        if event == cv2.EVENT_LBUTTONDOWN:
            x, y = mx, my
            last_click = (x, y)
            under = boxes_under(x, y)
            if not under:
                selected = None
            elif selected in under:
                selected = None
            else:
                selected = under[-1]

    cv2.namedWindow(WIN_NAME)
    cv2.setMouseCallback(WIN_NAME, on_mouse)

    while True:
        img_path = img_files[idx]
        img = cv2.imread(img_path)
        if img is None:
            print(f"Failed to load: {img_path}")
            idx = (idx + 1) % len(img_files)
            continue

        label_path = reload()
        h, w = img.shape[:2]
        for b in boxes:
            b["poly"] = to_pixels(b, w, h)
        selected = None
        last_click = None

        mode_tag = "EXACT" if use_exact else "MERGED"
        info = f"({idx+1}/{len(img_files)}) {os.path.basename(img_path)} [{mode_tag}] text={'ON' if show_text else 'OFF'}"
        cv2.putText(img, info, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 4)
        cv2.putText(img, info, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

        while True:
            display = img.copy()
            for b in boxes:
                draw_box(display, b, w, h, names, show_text, selected=(b is selected))

            if selected is not None:
                sel_name = (names or {}).get(selected["cls_id"]) or f"ID: {selected['cls_id']}"
                status = f"SELECTED box: class {selected['cls_id']} ({sel_name})  [M] reassign  [N] next"
            else:
                status = "Click a box to select it"
            cv2.putText(display, status, (10, display.shape[0] - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 4)
            cv2.putText(display, status, (10, display.shape[0] - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0) if selected is None else (0, 0, 0), 2)

            cv2.imshow(WIN_NAME, display)

            key = cv2.waitKey(25) & 0xFF
            if key == 0xFF:
                continue

            if key in [ord('d'), 83, 32]:
                idx = (idx + 1) % len(img_files)
                break
            elif key in [ord('a'), 81]:
                idx = (idx - 1) % len(img_files)
                break
            elif key in [ord('e'), 69]:
                use_exact = not use_exact
                label_path = reload()
                for b in boxes:
                    b["poly"] = to_pixels(b, w, h)
                selected = None
                print(f"Switched to {'exact' if use_exact else 'merged'} labels.")
            elif key in [ord('l'), 76]:
                show_text = not show_text
                print(f"Label text: {'ON' if show_text else 'OFF'}")
            elif key in [ord('n'), 78]:
                if last_click is not None:
                    under = boxes_under(*last_click)
                    if under:
                        if selected in under:
                            pos = under.index(selected)
                            selected = under[(pos + 1) % len(under)]
                        else:
                            selected = under[-1]
                        print(f"Selected box #{under.index(selected)} (class {selected['cls_id']})")
            elif key in [ord('m'), 77]:
                if selected is None:
                    print("No box selected. Click a box first.")
                else:
                    box = selected
                    new_id = pick_class(display, names, box["cls_id"])
                    if new_id is not None and new_id != box["cls_id"]:
                        old_id = rewrite_label_file(label_path, box, new_id)
                        old_name = (names or {}).get(old_id, old_id)
                        new_name = (names or {}).get(new_id, new_id)
                        print(f"{os.path.basename(label_path)}: class {old_name} -> {new_name}")
                    elif new_id is not None:
                        print("Same class; nothing changed.")
            elif key in [ord('q'), 27]:
                cv2.destroyAllWindows()
                return

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
