#!/usr/bin/env python3
"""Interactive map and RGB/thermal fusion viewer."""

from __future__ import annotations

import argparse
import json
import sys
import threading
from pathlib import Path

import numpy as np
from PIL import Image, ImageTk

try:
    import tkinter as tk
    from tkinter import ttk
except ImportError as error:  # pragma: no cover - depends on the host system
    raise SystemExit("Tkinter is required to run the map interface") from error


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SEASONS = ("winter", "spring", "summer", "autumn")
DAYTIMES = ("morning", "day", "evening")
CONDITIONS = tuple(f"{season}_{daytime}" for season in SEASONS for daytime in DAYTIMES)
WORLD_X = (5.0, 205.0)
WORLD_Y = (-100.0, 100.0)


def load_manifest(path: Path) -> list[dict]:
    with path.open() as stream:
        manifest = json.load(stream)
    if not isinstance(manifest, list):
        raise ValueError(f"Manifest must contain a JSON list: {path}")
    return manifest


def sequence_position(sequence: dict) -> tuple[float, float] | None:
    for frame in sequence.get("frames", []):
        position = frame.get("position")
        if position and len(position) >= 2:
            return float(position[0]), float(position[1])
    position = sequence.get("position")
    if position and len(position) >= 2:
        return float(position[0]), float(position[1])
    return None


class MapViewer:
    def __init__(self, root: tk.Tk, args: argparse.Namespace):
        self.root = root
        self.args = args
        self.map_dir = args.map_dir
        self.manifest = load_manifest(args.manifest)
        self.sequences = [
            sequence
            for sequence in self.manifest
            if sequence_position(sequence) is not None
        ]
        if not self.sequences:
            raise ValueError("The manifest contains no sequences with coordinates")

        self.season = tk.StringVar(value="summer")
        self.daytime = tk.StringVar(value="day")
        self.prediction_enabled = tk.BooleanVar(value=False)
        self.current_only = tk.BooleanVar(value=False)
        self.status = tk.StringVar(value="Click the map to select the closest point")
        self.selection: dict | None = None
        self.map_photo: ImageTk.PhotoImage | None = None
        self.map_source_size = (0, 0)
        self.map_display_size = (0, 0)
        self.map_zoom = 1.0
        self.map_center = (0.5, 0.5)
        self.marker_id: int | None = None
        self.image_cache: dict[Path, Image.Image] = {}
        self.display_photos: list[ImageTk.PhotoImage] = []
        self.prediction_photo: ImageTk.PhotoImage | None = None
        self.model = None
        self.model_error: str | None = None
        self.inference_generation = 0

        self._build_ui()
        self._load_map()
        self._refresh_selection()

    @property
    def condition(self) -> str:
        return f"{self.season.get()}_{self.daytime.get()}"

    def _build_ui(self) -> None:
        self.root.title("RGB/Thermal Map Viewer")
        self.root.geometry("1450x850")
        self.root.minsize(1050, 650)
        self.root.resizable(False, False)

        outer = ttk.Frame(self.root, padding=10)
        outer.pack(fill=tk.BOTH, expand=True)

        controls = ttk.Frame(outer)
        controls.pack(fill=tk.X, pady=(0, 8))

        season_controls = ttk.Frame(controls)
        season_controls.pack(side=tk.LEFT)
        ttk.Label(season_controls, text="Season:").pack(side=tk.LEFT, padx=(0, 6))
        for season in SEASONS:
            ttk.Radiobutton(
                season_controls,
                text=season.title(),
                value=season,
                variable=self.season,
                command=self._condition_changed,
            ).pack(side=tk.LEFT, padx=3)

        daytime_controls = ttk.Frame(controls)
        daytime_controls.pack(side=tk.LEFT, padx=(18, 0))
        ttk.Label(daytime_controls, text="Daytime:").pack(side=tk.LEFT, padx=(0, 6))
        for daytime in DAYTIMES:
            ttk.Radiobutton(
                daytime_controls,
                text=daytime.title(),
                value=daytime,
                variable=self.daytime,
                command=self._condition_changed,
            ).pack(side=tk.LEFT, padx=3)

        ttk.Checkbutton(
            controls,
            text="Show landmine prediction",
            variable=self.prediction_enabled,
            command=self._prediction_changed,
        ).pack(side=tk.RIGHT, padx=(20, 3))

        ttk.Checkbutton(
            controls,
            text="Use only current RGB + thermal",
            variable=self.current_only,
            command=self._prediction_mode_changed,
        ).pack(side=tk.RIGHT, padx=(8, 0))

        zoom_controls = ttk.Frame(controls)
        zoom_controls.pack(side=tk.RIGHT, padx=(8, 0))
        ttk.Label(zoom_controls, text="Map zoom:").pack(side=tk.LEFT, padx=(0, 4))
        ttk.Button(zoom_controls, text="-", width=3, command=lambda: self._zoom_map(0.8)).pack(
            side=tk.LEFT, padx=1
        )
        ttk.Button(zoom_controls, text="+", width=3, command=lambda: self._zoom_map(1.25)).pack(
            side=tk.LEFT, padx=1
        )
        ttk.Button(zoom_controls, text="Reset", command=self._reset_zoom).pack(
            side=tk.LEFT, padx=(3, 0)
        )

        self.map_frame = ttk.LabelFrame(outer, text="Overall map")
        self.map_frame.configure(width=785, height=780)
        self.map_frame.grid_propagate(False)
        self.map_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=False, padx=(0, 8))
        self.map_frame.rowconfigure(0, weight=1)
        self.map_frame.columnconfigure(0, weight=1)
        self.map_canvas = tk.Canvas(self.map_frame, background="#182019", highlightthickness=0)
        self.map_canvas.grid(row=0, column=0, sticky="nsew")
        self.map_canvas.bind("<Button-1>", self._map_clicked)
        self.map_canvas.bind("<MouseWheel>", self._map_wheel)
        self.map_canvas.bind("<Button-4>", lambda event: self._zoom_map(1.25, event))
        self.map_canvas.bind("<Button-5>", lambda event: self._zoom_map(0.8, event))
        self.map_canvas.bind("<Configure>", self._map_resized)

        self.detail_frame = ttk.LabelFrame(outer, text="Selected point")
        self.detail_frame.configure(width=635, height=780)
        self.detail_frame.grid_propagate(False)
        self.detail_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=False)
        self.detail_frame.columnconfigure(0, weight=1)
        self.detail_frame.columnconfigure(1, weight=1)
        self.detail_frame.rowconfigure(1, weight=1)
        self.detail_frame.rowconfigure(2, weight=1)

        self.point_label = ttk.Label(
            self.detail_frame, textvariable=self.status, justify=tk.LEFT, wraplength=560
        )
        self.point_label.grid(row=0, column=0, columnspan=2, sticky="ew", padx=8, pady=8)

        self.rgb_panel = self._make_image_panel("RGB image", 0, 1)
        self.thermal_panel = self._make_image_panel("Thermal image", 1, 1)
        self.prediction_panel = self._make_image_panel("Fusion prediction", 0, 2)
        self.overlay_panel = self._make_image_panel("Prediction overlay (RGB)", 1, 2)

    def _make_image_panel(
        self, title: str, column: int, row: int, colspan: int = 1
    ) -> ttk.Label:
        frame = ttk.LabelFrame(self.detail_frame, text=title)
        if colspan == 2:
            frame.configure(width=620, height=370)
        else:
            frame.configure(width=300, height=350)
        frame.grid_propagate(False)
        frame.grid(row=row, column=column, columnspan=colspan, sticky="nsew", padx=5, pady=5)
        frame.rowconfigure(0, weight=1)
        frame.columnconfigure(0, weight=1)
        label = ttk.Label(frame, anchor=tk.CENTER, text="No point selected")
        label.grid(row=0, column=0, sticky="nsew")
        return label

    def _map_path(self) -> Path:
        return self.map_dir / f"{self.condition}.png"

    def _load_map(self) -> None:
        path = self._map_path()
        if not path.exists():
            raise FileNotFoundError(f"Map image not found: {path}")
        with Image.open(path) as source:
            image = source.convert("RGB")
        self.map_source_size = image.size
        self._set_map_image(image)

    def _set_map_image(self, image: Image.Image) -> None:
        width = max(self.map_canvas.winfo_width(), 1)
        height = max(self.map_canvas.winfo_height(), 1)
        scale = min(width / image.width, height / image.height) * self.map_zoom
        display_size = (
            max(1, round(image.width * scale)),
            max(1, round(image.height * scale)),
        )
        displayed = image.resize(display_size, Image.Resampling.LANCZOS)
        self.map_display_size = display_size
        self.map_photo = ImageTk.PhotoImage(displayed)
        self.map_canvas.delete("all")
        x = round(width / 2 - self.map_center[0] * display_size[0])
        y = round(height / 2 - self.map_center[1] * display_size[1])
        self.map_canvas.create_image(x, y, image=self.map_photo, anchor=tk.NW)
        self.map_canvas.image_origin = (x, y)
        self._draw_marker()

    def _zoom_map(self, factor: float, event: tk.Event | None = None) -> None:
        old_zoom = self.map_zoom
        new_zoom = min(4.0, max(1.0, old_zoom * factor))
        if new_zoom == old_zoom:
            return

        if event is not None and self.map_display_size:
            origin_x, origin_y = getattr(self.map_canvas, "image_origin", (0, 0))
            point_u = (event.x - origin_x) / self.map_display_size[0]
            point_v = (event.y - origin_y) / self.map_display_size[1]
            canvas_center_x = self.map_canvas.winfo_width() / 2
            canvas_center_y = self.map_canvas.winfo_height() / 2
            scale_change = new_zoom / old_zoom
            new_width = self.map_display_size[0] * scale_change
            new_height = self.map_display_size[1] * scale_change
            center_u = point_u - (event.x - canvas_center_x) / new_width
            center_v = point_v - (event.y - canvas_center_y) / new_height
            visible_half_u = self.map_canvas.winfo_width() / (2 * new_width)
            visible_half_v = self.map_canvas.winfo_height() / (2 * new_height)
            self.map_center = (
                min(1 - visible_half_u, max(visible_half_u, center_u)),
                min(1 - visible_half_v, max(visible_half_v, center_v)),
            )

        self.map_zoom = new_zoom
        self._load_map()

    def _reset_zoom(self) -> None:
        self.map_zoom = 1.0
        self.map_center = (0.5, 0.5)
        self._load_map()

    def _map_wheel(self, event: tk.Event) -> str:
        if getattr(event, "delta", 0) > 0:
            self._zoom_map(1.25, event)
        elif getattr(event, "delta", 0) < 0:
            self._zoom_map(0.8, event)
        return "break"

    def _map_resized(self, _event=None) -> None:
        path = self._map_path()
        if path.exists():
            with Image.open(path) as source:
                self._set_map_image(source.convert("RGB"))

    def _map_clicked(self, event: tk.Event) -> None:
        origin_x, origin_y = getattr(self.map_canvas, "image_origin", (0, 0))
        display_width, display_height = self.map_display_size
        if not display_width or not display_height:
            return
        pixel_x = event.x - origin_x
        pixel_y = event.y - origin_y
        if not (0 <= pixel_x < display_width and 0 <= pixel_y < display_height):
            return
        # The map's vertical axis is world X; its horizontal axis is world Y.
        x = WORLD_X[0] + pixel_y / display_height * (WORLD_X[1] - WORLD_X[0])
        y = WORLD_Y[0] + pixel_x / display_width * (WORLD_Y[1] - WORLD_Y[0])
        self.selection = min(
            self.sequences,
            key=lambda sequence: self._distance_squared(sequence, x, y),
        )
        self._refresh_selection()

    @staticmethod
    def _distance_squared(sequence: dict, x: float, y: float) -> float:
        position = sequence_position(sequence)
        assert position is not None
        return (position[0] - x) ** 2 + (position[1] - y) ** 2

    def _condition_changed(self) -> None:
        self._load_map()
        self._refresh_selection()

    def _prediction_changed(self) -> None:
        if not self.prediction_enabled.get():
            self.prediction_photo = None
            self._clear_prediction()
            return
        if self.selection is None:
            self.prediction_enabled.set(False)
            self.status.set("Select a point on the map before enabling prediction")
            return
        self._start_prediction()

    def _prediction_mode_changed(self) -> None:
        if self.prediction_enabled.get() and self.selection is not None:
            self._start_prediction()

    def _refresh_selection(self) -> None:
        self._draw_marker()
        if self.selection is None:
            self.status.set("Click the map to select the closest point")
            self._set_label(self.rgb_panel, None, "No point selected")
            self._set_label(self.thermal_panel, None, "No point selected")
            self._clear_prediction()
            return

        position = sequence_position(self.selection)
        assert position is not None
        self.status.set(
            f"Sequence: {self.selection.get('sequence_id', 'unknown')}\n"
            f"World position: x={position[0]:.2f}, y={position[1]:.2f}\n"
            f"Condition: {self.condition}"
        )
        frame = self._frame_for_condition(self.selection, self.condition)
        self._set_label(self.rgb_panel, self._frame_path(frame, "rgb_path"), "RGB image unavailable")
        self._set_label(
            self.thermal_panel,
            self._frame_path(frame, "thermal_image_path"),
            "Thermal image unavailable",
        )
        if self.prediction_enabled.get():
            self._start_prediction()
        else:
            self._clear_prediction()

    @staticmethod
    def _frame_for_condition(sequence: dict, condition: str) -> dict:
        for frame in sequence.get("frames", []):
            if frame.get("condition") == condition:
                return frame
        return {}

    @staticmethod
    def _frame_path(frame: dict, key: str) -> Path | None:
        value = frame.get(key)
        return Path(value) if value else None

    def _set_label(self, label: ttk.Label, path: Path | None, fallback: str) -> None:
        if path is None or not path.exists():
            label.configure(image="", text=fallback)
            label.image = None
            return
        image = self._read_image(path)
        self._set_label_image(label, image)

    def _set_label_image(self, label: ttk.Label, image: Image.Image) -> None:
        width = max(label.winfo_width(), 280)
        height = max(label.winfo_height(), 250)
        displayed = image.copy()
        displayed.thumbnail((width - 12, height - 12), Image.Resampling.LANCZOS)
        photo = ImageTk.PhotoImage(displayed)
        label.configure(image=photo, text="")
        label.image = photo
        self.display_photos.append(photo)
        if len(self.display_photos) > 20:
            self.display_photos = self.display_photos[-10:]

    def _read_image(self, path: Path) -> Image.Image:
        cached = self.image_cache.get(path)
        if cached is None:
            with Image.open(path) as source:
                cached = source.convert("RGB")
            self.image_cache[path] = cached
        return cached

    def _draw_marker(self) -> None:
        if self.marker_id is not None:
            self.map_canvas.delete(self.marker_id)
            self.marker_id = None
        if self.selection is None or not self.map_display_size:
            return
        position = sequence_position(self.selection)
        assert position is not None
        origin_x, origin_y = getattr(self.map_canvas, "image_origin", (0, 0))
        width, height = self.map_display_size
        pixel_x = (position[1] - WORLD_Y[0]) / (WORLD_Y[1] - WORLD_Y[0]) * width
        pixel_y = (position[0] - WORLD_X[0]) / (WORLD_X[1] - WORLD_X[0]) * height
        x, y = origin_x + pixel_x, origin_y + pixel_y
        radius = 8
        self.marker_id = self.map_canvas.create_oval(
            x - radius,
            y - radius,
            x + radius,
            y + radius,
            fill="#ff352f",
            outline="white",
            width=2,
        )

    def _clear_prediction(self) -> None:
        self.prediction_panel.configure(image="", text="Prediction disabled")
        self.prediction_panel.image = None
        self.overlay_panel.configure(image="", text="Prediction disabled")
        self.overlay_panel.image = None

    def _start_prediction(self) -> None:
        if self.selection is None:
            return
        self.inference_generation += 1
        generation = self.inference_generation
        condition = self.condition
        current_only = self.current_only.get()
        self.prediction_panel.configure(image="", text="Running fusion model...")
        self.prediction_panel.image = None
        self.overlay_panel.configure(image="", text="Running fusion model...")
        self.overlay_panel.image = None
        thread = threading.Thread(
            target=self._predict_worker,
            args=(self.selection, generation, condition, current_only),
            daemon=True,
        )
        thread.start()

    def _predict_worker(
        self, sequence: dict, generation: int, condition: str, current_only: bool
    ) -> None:
        try:
            if self.model is None:
                self._load_model()
            import torch
            from phase2_rgb_thermal_image_fusion.data import ImageFusionSequenceDataset

            dataset = ImageFusionSequenceDataset([sequence], tile_size=self.args.tile_size)
            item = dataset[0]
            device = self.args.device
            batch = {
                key: value.unsqueeze(0).to(device)
                for key, value in item.items()
                if torch.is_tensor(value)
            }
            if current_only:
                condition_index = CONDITIONS.index(condition)
                for sequence_key, availability_key in (
                    ("rgb_sequence", "rgb_available"),
                    ("thermal_image", "thermal_available"),
                ):
                    selected_frame = batch[sequence_key][:, condition_index].clone()
                    batch[sequence_key].zero_()
                    batch[sequence_key][:, condition_index] = selected_frame
                    batch[availability_key].zero_()
                    batch[availability_key][:, condition_index] = True
            with torch.inference_mode():
                logits, _details = self.model(
                    batch["rgb_sequence"],
                    batch["thermal_image"],
                    batch["rgb_available"],
                    batch["thermal_available"],
                    freeze_branches=True,
                )
            probability = torch.sigmoid(logits[0, 0]).cpu().numpy()
            image = self._probability_image(probability)
            overlay = self._prediction_overlay(sequence, probability)
            self.root.after(0, self._show_prediction, image, overlay, generation, None)
        except Exception as error:  # Keep inference failures inside the GUI.
            self.root.after(0, self._show_prediction, None, None, generation, str(error))

    def _load_model(self) -> None:
        import torch
        from phase2_rgb_thermal_image_fusion.models import ImageFusionSegmenter
        from phase2_rgb_thermal_image_fusion.scripts.infer_image_fusion import load_compatible

        model = ImageFusionSegmenter().to(self.args.device).eval()
        payload = torch.load(
            self.args.checkpoint, map_location=self.args.device, weights_only=False
        )
        load_compatible(model, payload)
        self.model = model

    def _probability_image(self, probability: np.ndarray) -> Image.Image:
        import matplotlib.cm as cm

        rgba = cm.get_cmap("inferno")(np.clip(probability, 0, 1))
        image = Image.fromarray((rgba[..., :3] * 255).astype(np.uint8))
        return image.resize((self.args.display_size, self.args.display_size), Image.Resampling.BILINEAR)

    def _prediction_overlay(self, sequence: dict, probability: np.ndarray) -> Image.Image:
        frame = self._frame_for_condition(sequence, self.condition)
        rgb_path = self._frame_path(frame, "rgb_path")
        if rgb_path is None or not rgb_path.exists():
            return Image.fromarray(
                np.zeros((self.args.display_size, self.args.display_size, 3), dtype=np.uint8)
            )
        base = np.asarray(self._read_image(rgb_path).convert("RGB"), dtype=np.float32)
        resized_probability = np.asarray(
            Image.fromarray((np.clip(probability, 0, 1) * 255).astype(np.uint8)).resize(
                (base.shape[1], base.shape[0]), Image.Resampling.BILINEAR
            ),
            dtype=np.float32,
        ) / 255.0
        mask = resized_probability >= 0.5
        base[mask] = base[mask] * 0.55 + np.asarray((255, 45, 45), dtype=np.float32) * 0.45
        return Image.fromarray(np.clip(base, 0, 255).astype(np.uint8))

    def _show_prediction(
        self,
        image: Image.Image | None,
        overlay: Image.Image | None,
        generation: int,
        error: str | None,
    ) -> None:
        if generation != self.inference_generation or not self.prediction_enabled.get():
            return
        if error:
            self.prediction_panel.configure(image="", text=f"Prediction failed:\n{error}")
            self.prediction_panel.image = None
            self.overlay_panel.configure(image="", text="Prediction unavailable")
            self.overlay_panel.image = None
            return
        assert image is not None and overlay is not None
        self._set_label_image(self.prediction_panel, image)
        self._set_label_image(self.overlay_panel, overlay)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "phase2_rgb_thermal_image_fusion/artifacts/fusion_main_manifest.json",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=ROOT / "phase2_rgb_thermal_image_fusion/artifacts/run_fusion_main/epochs/epoch_0039/image_fusion_checkpoint.pt",
    )
    parser.add_argument("--map-dir", type=Path, default=ROOT / "map/images")
    parser.add_argument("--tile-size", type=int, default=640)
    parser.add_argument("--display-size", type=int, default=420)
    parser.add_argument(
        "--device", default="cuda" if _cuda_available() else "cpu"
    )
    return parser.parse_args()


def _cuda_available() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except ImportError:
        return False


def main() -> None:
    args = parse_args()
    for path in (args.manifest, args.map_dir, args.checkpoint):
        if not path.exists():
            raise SystemExit(f"Required path does not exist: {path}")
    if args.tile_size <= 0 or args.display_size <= 0:
        raise SystemExit("tile size and display size must be positive")
    root = tk.Tk()
    try:
        MapViewer(root, args)
        root.mainloop()
    except Exception as error:
        root.destroy()
        raise SystemExit(str(error)) from error


if __name__ == "__main__":
    main()
