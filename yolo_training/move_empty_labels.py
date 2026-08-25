import argparse
import os
import shutil

EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")


def is_empty_label(path):
    if not os.path.exists(path):
        return True
    with open(path, "r") as f:
        for line in f:
            if line.strip():
                return False
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Move images with empty/missing labels to a separate dataset."
    )
    parser.add_argument(
        "dataset_dir",
        nargs="?",
        default="/media/geekscode/Developer1/Projects/gd-autopilot/datasets/dataset_auto_ultimate_test_20260824_203515",
        help="Path to dataset directory containing images/, labels/ and exact_labels/",
    )
    parser.add_argument(
        "--suffix",
        default="_empty",
        help="Suffix appended to dataset dir name for the target (default: '_empty')",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only list what would be moved, without moving anything",
    )
    args = parser.parse_args()

    src = args.dataset_dir.rstrip("/")
    if not os.path.isdir(src):
        print(f"Error: dataset dir not found: {src}")
        return

    images_dir = os.path.join(src, "images")
    labels_dir = os.path.join(src, "labels")
    exact_dir = os.path.join(src, "exact_labels")

    dst = src + args.suffix
    dst_images = os.path.join(dst, "images")
    dst_labels = os.path.join(dst, "labels")
    dst_exact = os.path.join(dst, "exact_labels")

    if not os.path.isdir(images_dir):
        print(f"Error: images directory not found: {images_dir}")
        return

    if not args.dry_run:
        for folder in (dst_images, dst_labels, dst_exact):
            os.makedirs(folder, exist_ok=True)

    to_move = []
    for fname in sorted(os.listdir(images_dir)):
        if not fname.lower().endswith(EXTENSIONS):
            continue
        stem = os.path.splitext(fname)[0]
        label_path = os.path.join(labels_dir, stem + ".txt")
        if is_empty_label(label_path):
            to_move.append((stem, fname))

    print(f"Dataset: {src}")
    print(f"Images to move (empty labels): {len(to_move)}")
    print(f"Target: {dst}")
    print()

    moved_files = 0
    for stem, fname in to_move:
        groups = {
            "images": (os.path.join(images_dir, fname), os.path.join(dst_images, fname)),
            "labels": (os.path.join(labels_dir, stem + ".txt"), os.path.join(dst_labels, stem + ".txt")),
            "exact": (os.path.join(exact_dir, stem + ".txt"), os.path.join(dst_exact, stem + ".txt")),
        }
        for kind, (s, d) in groups.items():
            if os.path.exists(s):
                if args.dry_run:
                    print(f"  would move: {s} -> {d}")
                else:
                    os.replace(s, d)
                moved_files += 1

    print()
    if args.dry_run:
        print(f"Dry run complete: {moved_files} files would be moved.")
    else:
        for folder in (dst_images, dst_labels, dst_exact):
            if os.path.isdir(folder) and not os.listdir(folder):
                os.rmdir(folder)
        for name in ("classes.txt",):
            for kind, sdir, ddir in (("labels", labels_dir, dst_labels), ("exact", exact_dir, dst_exact)):
                spath = os.path.join(sdir, name)
                dpath = os.path.join(ddir, name)
                if os.path.exists(spath) and not os.path.exists(dpath):
                    shutil.copy2(spath, dpath)
                    print(f"  copied: {spath} -> {dpath}")
        print(f"Moved {moved_files} files. Empty-label images: {len(to_move)}")
        print(f"New dataset: {dst}")


if __name__ == "__main__":
    main()
