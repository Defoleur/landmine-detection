import json
import os
import shutil
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext

EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")
SUBDIRS = ("images", "labels", "exact_labels")


class MergeDatasetsApp:
    def __init__(self, root):
        root.title("Merge / Unmerge Datasets")
        root.geometry("720x580")
        root.resizable(True, True)

        nb = tk.Frame(root)
        nb.pack(fill="x", padx=6, pady=(6, 0))
        self.tab_merge_btn = tk.Button(nb, text="Merge", font=("Arial", 10, "bold"))
        self.tab_merge_btn.pack(side="left", padx=(0, 2))
        self.tab_unmerge_btn = tk.Button(nb, text="Unmerge", font=("Arial", 10, "bold"))
        self.tab_unmerge_btn.pack(side="left")

        self.notebook = tk.Frame(root)
        self.notebook.pack(fill="both", expand=True, padx=6, pady=6)

        self.merge_frame = tk.Frame(self.notebook)
        self.unmerge_frame = tk.Frame(self.notebook)

        self._build_merge_tab()
        self._build_unmerge_tab()

        self.tab_merge_btn.config(command=lambda: self._show_tab("merge"))
        self.tab_unmerge_btn.config(command=lambda: self._show_tab("unmerge"))
        self._show_tab("merge")

    def _show_tab(self, name):
        self.merge_frame.pack_forget()
        self.unmerge_frame.pack_forget()
        if name == "merge":
            self.merge_frame.pack(fill="both", expand=True)
            self.tab_merge_btn.config(relief="sunken")
            self.tab_unmerge_btn.config(relief="raised")
        else:
            self.unmerge_frame.pack(fill="both", expand=True)
            self.tab_merge_btn.config(relief="raised")
            self.tab_unmerge_btn.config(relief="sunken")

    def _build_merge_tab(self):
        f = self.merge_frame

        tk.Label(f, text="Source datasets (one per line):", anchor="w").pack(fill="x")
        self.merge_sources = tk.Text(f, height=6, width=80)
        self.merge_sources.pack(fill="x", pady=(0, 4))

        browse_row = tk.Frame(f)
        browse_row.pack(fill="x", pady=(0, 4))
        tk.Button(browse_row, text="Browse dirs...", command=self._browse_merge_sources).pack(side="left")
        tk.Button(browse_row, text="Clear", command=lambda: self.merge_sources.delete("1.0", "end")).pack(side="left", padx=4)

        out_row = tk.Frame(f)
        out_row.pack(fill="x", pady=(0, 4))
        tk.Label(out_row, text="Output dir:").pack(side="left")
        self.merge_output = tk.Entry(out_row, width=60)
        self.merge_output.pack(side="left", fill="x", expand=True, padx=4)
        tk.Button(out_row, text="Browse...", command=self._browse_merge_output).pack(side="left")

        self.merge_btn = tk.Button(f, text="Merge", bg="#4a9", fg="white", font=("Arial", 10, "bold"), command=self._run_merge)
        self.merge_btn.pack(pady=4)

        tk.Label(f, text="Log:", anchor="w").pack(fill="x")
        self.merge_log = scrolledtext.ScrolledText(f, height=12, state="disabled", bg="#1e1e1e", fg="#cccccc", insertbackground="white")
        self.merge_log.pack(fill="both", expand=True)

    def _build_unmerge_tab(self):
        f = self.unmerge_frame

        row1 = tk.Frame(f)
        row1.pack(fill="x", pady=(0, 4))
        tk.Label(row1, text="Merged dataset dir:").pack(side="left")
        self.unmerge_merged = tk.Entry(row1, width=60)
        self.unmerge_merged.pack(side="left", fill="x", expand=True, padx=4)
        tk.Button(row1, text="Browse...", command=self._browse_unmerge_merged).pack(side="left")

        row2 = tk.Frame(f)
        row2.pack(fill="x", pady=(0, 4))
        tk.Label(row2, text="Manifest JSON:").pack(side="left")
        self.unmerge_manifest = tk.Entry(row2, width=60)
        self.unmerge_manifest.pack(side="left", fill="x", expand=True, padx=4)
        tk.Button(row2, text="Browse...", command=self._browse_unmerge_manifest).pack(side="left")

        self.unmerge_cleanup = tk.BooleanVar(value=False)
        tk.Checkbutton(f, text="Remove merged dir after unmerge", variable=self.unmerge_cleanup).pack(anchor="w", pady=(0, 4))

        self.unmerge_btn = tk.Button(f, text="Unmerge", bg="#a44", fg="white", font=("Arial", 10, "bold"), command=self._run_unmerge)
        self.unmerge_btn.pack(pady=4)

        tk.Label(f, text="Log:", anchor="w").pack(fill="x")
        self.unmerge_log = scrolledtext.ScrolledText(f, height=14, state="disabled", bg="#1e1e1e", fg="#cccccc", insertbackground="white")
        self.unmerge_log.pack(fill="both", expand=True)

    def _browse_merge_sources(self):
        d = filedialog.askdirectory(title="Select source dataset")
        if d:
            cur = self.merge_sources.get("1.0", "end").strip()
            if cur:
                self.merge_sources.insert("end", "\n" + d)
            else:
                self.merge_sources.insert("1.0", d)

    def _browse_merge_output(self):
        d = filedialog.askdirectory(title="Select output directory")
        if d:
            self.merge_output.delete(0, "end")
            self.merge_output.insert(0, d)

    def _browse_unmerge_merged(self):
        d = filedialog.askdirectory(title="Select merged dataset directory")
        if d:
            self.unmerge_merged.delete(0, "end")
            self.unmerge_merged.insert(0, d)

    def _browse_unmerge_manifest(self):
        f = filedialog.askopenfilename(title="Select manifest.json", filetypes=[("JSON", "*.json"), ("All", "*.*")])
        if f:
            self.unmerge_manifest.delete(0, "end")
            self.unmerge_manifest.insert(0, f)

    def _log(self, log_widget, msg):
        log_widget.config(state="normal")
        log_widget.insert("end", msg + "\n")
        log_widget.see("end")
        log_widget.config(state="disabled")

    def _run_merge(self):
        sources = [s.strip() for s in self.merge_sources.get("1.0", "end").strip().splitlines() if s.strip()]
        output = self.merge_output.get().strip()
        if not sources or not output:
            messagebox.showerror("Error", "Fill in source datasets and output dir.")
            return
        self.merge_btn.config(state="disabled")
        self.merge_log.config(state="normal")
        self.merge_log.delete("1.0", "end")
        self.merge_log.config(state="disabled")

        def worker():
            try:
                os.makedirs(output, exist_ok=True)
                manifest = {
                    "manifest_version": 1,
                    "merged_from": [os.path.abspath(s) for s in sources],
                    "images": {},
                }
                total = 0
                for src in sources:
                    src = src.rstrip("/")
                    src_images = os.path.join(src, "images")
                    if not os.path.isdir(src_images):
                        self._log(self.merge_log, f"Skipping {src} (no images/ dir)")
                        continue
                    self._log(self.merge_log, f"Merging: {src}")
                    for subdir in SUBDIRS:
                        s = os.path.join(src, subdir)
                        if os.path.isdir(s):
                            os.makedirs(os.path.join(output, subdir), exist_ok=True)
                    for fname in sorted(os.listdir(src_images)):
                        if not fname.lower().endswith(EXTENSIONS):
                            continue
                        stem = os.path.splitext(fname)[0]
                        dst_stem = fname
                        dst_img = os.path.join(output, "images", dst_stem)
                        if dst_stem in manifest["images"]:
                            self._log(self.merge_log, f"  Skip (dup): {dst_stem}")
                            continue
                        manifest["images"][dst_stem] = {
                            "source": os.path.abspath(src),
                            "original_name": fname,
                        }
                        if os.path.exists(dst_img):
                            self._log(self.merge_log, f"  Skip (exists): {dst_stem}")
                            continue
                        shutil.copy2(os.path.join(src_images, fname), dst_img)
                        for subdir in ("labels", "exact_labels"):
                            s_label = os.path.join(src, subdir, stem + ".txt")
                            d_label = os.path.join(output, subdir, dst_stem.replace(os.path.splitext(dst_stem)[1], ".txt"))
                            if os.path.exists(s_label):
                                os.makedirs(os.path.join(output, subdir), exist_ok=True)
                                shutil.copy2(s_label, d_label)
                        total += 1
                manifest_path = os.path.join(output, "manifest.json")
                with open(manifest_path, "w") as f:
                    json.dump(manifest, f, indent=2)
                self._log(self.merge_log, f"\nDone: {total} images merged into {output}")
                self._log(self.merge_log, f"Manifest: {manifest_path}")
            except Exception as e:
                self._log(self.merge_log, f"ERROR: {e}")
            finally:
                self.merge_btn.config(state="normal")

        threading.Thread(target=worker, daemon=True).start()

    def _run_unmerge(self):
        merged = self.unmerge_merged.get().strip()
        manifest_path = self.unmerge_manifest.get().strip()
        cleanup = self.unmerge_cleanup.get()
        if not merged or not manifest_path:
            messagebox.showerror("Error", "Fill in merged dir and manifest path.")
            return
        if not os.path.exists(manifest_path):
            messagebox.showerror("Error", f"Manifest not found: {manifest_path}")
            return
        self.unmerge_btn.config(state="disabled")
        self.unmerge_log.config(state="normal")
        self.unmerge_log.delete("1.0", "end")
        self.unmerge_log.config(state="disabled")

        def worker():
            try:
                with open(manifest_path) as f:
                    manifest = json.load(f)
                images_dir = os.path.join(merged, "images")
                moved = 0
                for dst_stem, info in manifest["images"].items():
                    source = info["source"]
                    orig_name = info["original_name"]
                    src_img = os.path.join(images_dir, dst_stem)
                    if not os.path.exists(src_img):
                        continue
                    for subdir in SUBDIRS:
                        os.makedirs(os.path.join(source, subdir), exist_ok=True)
                    shutil.move(src_img, os.path.join(source, "images", orig_name))
                    orig_stem = os.path.splitext(orig_name)[0]
                    for subdir in ("labels", "exact_labels"):
                        src_label = os.path.join(merged, subdir, dst_stem.replace(os.path.splitext(dst_stem)[1], ".txt"))
                        if os.path.exists(src_label):
                            shutil.move(src_label, os.path.join(source, subdir, orig_stem + ".txt"))
                    moved += 1
                    if moved % 1000 == 0:
                        self._log(self.unmerge_log, f"  ...{moved} moved")
                self._log(self.unmerge_log, f"Unmerged {moved} images.")
                if cleanup:
                    self._log(self.unmerge_log, "Cleaning up merged dir...")
                    for name in os.listdir(merged):
                        path = os.path.join(merged, name)
                        if os.path.isfile(path):
                            os.remove(path)
                        elif os.path.isdir(path):
                            shutil.rmtree(path)
                    os.rmdir(merged)
                    self._log(self.unmerge_log, "Removed merged directory.")
                self._log(self.unmerge_log, "Done.")
            except Exception as e:
                self._log(self.unmerge_log, f"ERROR: {e}")
            finally:
                self.unmerge_btn.config(state="normal")

        threading.Thread(target=worker, daemon=True).start()


def main():
    root = tk.Tk()
    MergeDatasetsApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
