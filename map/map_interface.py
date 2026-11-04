#!/usr/bin/env python3
"""Interactive map and RGB/thermal fusion viewer."""

from __future__ import annotations

import argparse
import json
import sys
import threading
from pathlib import Path

import numpy as np
from PIL import Image

try:
    from PyQt5.QtCore import QPoint, QPointF, QTimer, Qt, QThread, pyqtSignal
    from PyQt5.QtGui import QColor, QFont, QImage, QPainter, QPen, QPixmap, QPolygonF
    from PyQt5.QtWidgets import (
        QApplication,
        QCheckBox,
        QButtonGroup,
        QDialog,
        QDialogButtonBox,
        QFileDialog,
        QFormLayout,
        QGridLayout,
        QGroupBox,
        QHBoxLayout,
        QLabel,
        QLineEdit,
        QMainWindow,
        QMessageBox,
        QProgressBar,
        QPushButton,
        QRadioButton,
        QScrollArea,
        QSlider,
        QSizePolicy,
        QVBoxLayout,
        QWidget,
    )
except ImportError as error:  # pragma: no cover - depends on the host system
    raise SystemExit("PyQt5 is required to run the map interface") from error


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from phase1d_thermal_convlstm.sequence import CONDITIONS

SEASONS = ("winter", "spring", "summer", "autumn")
DAYTIMES = ("morning", "day", "evening")
CONDITIONS = tuple(f"{season}_{daytime}" for season in SEASONS for daytime in DAYTIMES)
WORLD_X = (5.0, 205.0)
WORLD_Y = (-100.0, 100.0)
DEFAULT_DATASET = ROOT / "datasets" / "dataset_full_map_20261019_181935"
DEFAULT_MAP_DIR = ROOT / "map" / "images_merged"
DEFAULT_CONFIG = ROOT / "map" / "map_interface_config.json"


def _config_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT / path


def load_interface_config(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open() as stream:
        config = json.load(stream)
    if not isinstance(config, dict):
        raise ValueError(f"Interface config must be a JSON object: {path}")
    return config


def _config_path_value(config: dict, key: str, fallback: Path) -> Path:
    value = config.get(key)
    return _config_path(value) if isinstance(value, str) and value else fallback


def _config_pair(config: dict, key: str, fallback: tuple[float, float] | None = None):
    value = config.get(key)
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return fallback
    try:
        return float(value[0]), float(value[1])
    except (TypeError, ValueError):
        return fallback


def _config_size(config: dict) -> tuple[int, int] | None:
    value = config.get("map_size")
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return None
    try:
        size = int(value[0]), int(value[1])
    except (TypeError, ValueError):
        return None
    return size if min(size) > 0 else None


def load_manifest(path: Path) -> tuple[list[dict], tuple[float, float], tuple[float, float], float, dict]:
    with path.open() as stream:
        manifest = json.load(stream)
    if not isinstance(manifest, dict) or not isinstance(manifest.get("captures"), list):
        raise ValueError(f"Manifest must be a full-map dataset manifest: {path}")
    bounds = manifest.get("map_bounds_xy")
    if not bounds or len(bounds) != 2:
        raise ValueError(f"Manifest has no valid map bounds: {path}")
    world_x = (float(bounds[0][0]), float(bounds[1][0]))
    world_y = (float(bounds[0][1]), float(bounds[1][1]))
    return (
        manifest["captures"],
        world_x,
        world_y,
        float(manifest.get("tile_size_m", 3.0)),
        manifest,
    )


def sequence_position(sequence: dict) -> tuple[float, float] | None:
    position = sequence.get("position")
    if position and len(position) >= 2:
        return float(position[0]), float(position[1])
    return None


def capture_corners(capture: dict, tile_size: float = 3.0) -> list[tuple[float, float]] | None:
    corners = capture.get("image_corners_world")
    if not isinstance(corners, dict):
        return None
    keys = ("top_left", "top_right", "bottom_right", "bottom_left")
    if isinstance(corners, dict) and not any(
        not isinstance(corners.get(key), list) or len(corners[key]) < 2 for key in keys
    ):
        return [(float(corners[key][0]), float(corners[key][1])) for key in keys]
    position = capture.get("position")
    if not isinstance(position, list) or len(position) < 2:
        return None
    center_x, center_y = map(float, position[:2])
    half_size = tile_size / 2.0
    return [
        (center_x + half_size, center_y - half_size),
        (center_x + half_size, center_y + half_size),
        (center_x - half_size, center_y + half_size),
        (center_x - half_size, center_y - half_size),
    ]


def pil_to_pixmap(image: Image.Image) -> QPixmap:
    rgb = image.convert("RGB")
    data = rgb.tobytes("raw", "RGB")
    qimage = QImage(data, rgb.width, rgb.height, rgb.width * 3, QImage.Format_RGB888)
    return QPixmap.fromImage(qimage.copy())


def _load_utae_model(checkpoint: Path, device):
    import torch
    from phase2_rgb_thermal_image_utae.models import AsyncUTAESegmenter

    model = AsyncUTAESegmenter().to(device).eval()
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(payload.get("model_state_dict", payload), strict=False)
    return model


def _utae_image_tensor(path: Path, tile_size: int, device):
    from phase2_rgb_thermal_image_fusion.data import imagenet_normalize
    with Image.open(path) as source:
        image = source.convert("RGB").resize(
            (tile_size, tile_size), Image.Resampling.BILINEAR
        )
    values = np.asarray(image, dtype=np.float32) / 255.0
    values = imagenet_normalize(values.transpose(2, 0, 1))
    import torch
    return torch.from_numpy(values).to(device)


def _utae_metadata(conditions, rgb_available, thermal_available, device):
    import torch
    from phase2_rgb_thermal_image_utae.manifest import (
        id_time_features,
        legacy_modality_ages,
    )

    season_ids = torch.tensor(
        [[SEASONS.index(condition.split("_")[0]) for condition in conditions]],
        dtype=torch.long,
    )
    daytime_ids = torch.tensor(
        [[DAYTIMES.index(condition.split("_")[1]) for condition in conditions]],
        dtype=torch.long,
    )
    rgb_ages, thermal_ages = legacy_modality_ages(
        season_ids,
        daytime_ids,
        rgb_available.cpu(),
        thermal_available.cpu(),
    )
    time_features = id_time_features(season_ids, daytime_ids).to(device)
    return {
        "rgb_time_features": time_features,
        "thermal_time_features": time_features,
        "rgb_ages": rgb_ages.to(device),
        "thermal_ages": thermal_ages.to(device),
    }


class MapCanvas(QWidget):
    map_clicked = pyqtSignal(float, float)
    zoom_changed = pyqtSignal(float, float, QPoint)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.source: QPixmap | None = None
        self.prediction_overlay: QPixmap | None = None
        self.prediction_overlay_opacity = 0.52
        self.source_size = (0, 0)
        self.zoom = 1.0
        self.center = (0.5, 0.5)
        self.viewport_size = (400, 400)
        self.selected_corners: list[tuple[float, float]] | None = None
        self.world_x = WORLD_X
        self.world_y = WORLD_Y
        self.rotated_180 = False
        self.setMinimumSize(400, 400)
        self.setMouseTracking(True)
        self.setBackgroundRole(self.backgroundRole())

    def set_map_image(self, image: Image.Image, map_size: tuple[int, int] | None = None) -> None:
        if map_size and image.size != map_size:
            image = image.resize(map_size, Image.Resampling.LANCZOS)
        self.source = pil_to_pixmap(image)
        self.source_size = image.size
        self.update()

    def set_prediction_overlay(self, image: Image.Image | None) -> None:
        self.prediction_overlay = pil_to_pixmap(image) if image is not None else None
        self.update()

    def set_prediction_overlay_opacity(self, value: int) -> None:
        self.prediction_overlay_opacity = max(0, min(100, value)) / 100.0
        self.update()

    def set_map_orientation(self, rotated_180: bool) -> None:
        self.rotated_180 = rotated_180
        self.update()

    def clear_map(self) -> None:
        self.source = None
        self.prediction_overlay = None
        self.source_size = (0, 0)
        self.set_selected_corners(None)

    def set_world_bounds(self, world_x: tuple[float, float], world_y: tuple[float, float]) -> None:
        self.world_x = world_x
        self.world_y = world_y

    def set_selected_corners(self, corners: list[tuple[float, float]] | None) -> None:
        self.selected_corners = corners
        self.update()

    def reset_zoom(self) -> None:
        old_zoom = self.zoom
        self.zoom = 1.0
        self.center = (0.5, 0.5)
        self.zoom_changed.emit(old_zoom, self.zoom, QPoint(-1, -1))
        self._sync_content_size()
        self.update()

    def zoom_by(self, factor: float, cursor: QPoint | None = None) -> None:
        if self.source is None:
            return
        old_zoom = self.zoom
        new_zoom = min(8.0, max(1.0, old_zoom * factor))
        if new_zoom == old_zoom:
            return
        self.zoom = new_zoom
        self.zoom_changed.emit(old_zoom, new_zoom, cursor or QPoint(-1, -1))
        self._sync_content_size()
        self.update()

    def set_viewport_size(self, width: int, height: int) -> None:
        self.viewport_size = (max(width, 1), max(height, 1))
        self._sync_content_size()

    def _sync_content_size(self) -> None:
        width, height = self.viewport_size
        if self.source is None:
            self.resize(max(400, width), max(400, height))
            return
        display_width, display_height = self._display_size()
        self.resize(display_width, display_height)

    def _display_size(self, zoom: float | None = None) -> tuple[int, int]:
        if self.source is None:
            return (0, 0)
        zoom = self.zoom if zoom is None else zoom
        viewport_width, viewport_height = self.viewport_size
        scale = min(viewport_width / self.source.width(), viewport_height / self.source.height()) * zoom
        return (max(1, round(self.source.width() * scale)), max(1, round(self.source.height() * scale)))

    def _origin(self, display_size: tuple[int, int]) -> tuple[int, int]:
        return (
            round(self.width() / 2 - self.center[0] * display_size[0]),
            round(self.height() / 2 - self.center[1] * display_size[1]),
        )

    def _image_coordinates(self, position: QPoint) -> tuple[float, float] | None:
        display_size = self._display_size()
        if not display_size:
            return None
        origin = self._origin(display_size)
        pixel_x = position.x() - origin[0]
        pixel_y = position.y() - origin[1]
        if not (0 <= pixel_x < display_size[0] and 0 <= pixel_y < display_size[1]):
            return None
        x_fraction = pixel_y / display_size[1]
        y_fraction = pixel_x / display_size[0]
        if self.rotated_180:
            x_fraction = 1.0 - x_fraction
            y_fraction = 1.0 - y_fraction
        x = self.world_x[0] + x_fraction * (self.world_x[1] - self.world_x[0])
        y = self.world_y[0] + y_fraction * (self.world_y[1] - self.world_y[0])
        return x, y

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#171717"))
        if self.source is None:
            return
        display_size = self._display_size()
        origin = self._origin(display_size)
        scaled = self.source.scaled(*display_size, Qt.IgnoreAspectRatio, Qt.SmoothTransformation)
        painter.drawPixmap(*origin, scaled)
        if self.prediction_overlay is not None:
            overlay = self.prediction_overlay.scaled(
                *display_size, Qt.IgnoreAspectRatio, Qt.SmoothTransformation
            )
            painter.setOpacity(self.prediction_overlay_opacity)
            painter.drawPixmap(*origin, overlay)
            painter.setOpacity(1.0)
        if self.selected_corners is None:
            return
        painter.setRenderHint(QPainter.Antialiasing)
        polygon = QPolygonF()
        for x_world, y_world in self.selected_corners:
            x_fraction = (y_world - self.world_y[0]) / (self.world_y[1] - self.world_y[0])
            y_fraction = (x_world - self.world_x[0]) / (self.world_x[1] - self.world_x[0])
            if self.rotated_180:
                x_fraction = 1.0 - x_fraction
                y_fraction = 1.0 - y_fraction
            x = origin[0] + x_fraction * display_size[0]
            y = origin[1] + y_fraction * display_size[1]
            polygon.append(QPointF(x, y))
        painter.setPen(QPen(QColor("#f59e0b"), 2))
        painter.setBrush(QColor(245, 158, 11, 45))
        painter.drawPolygon(polygon)

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.LeftButton:
            coordinates = self._image_coordinates(event.pos())
            if coordinates:
                self.map_clicked.emit(*coordinates)
        super().mousePressEvent(event)

    def wheelEvent(self, event) -> None:
        self.zoom_by(1.25 if event.angleDelta().y() > 0 else 0.8, event.pos())
        event.accept()


class MapScrollArea(QScrollArea):
    def __init__(self, canvas: MapCanvas, parent=None):
        super().__init__(parent)
        self.canvas = canvas
        self.canvas.zoom_changed.connect(self._preserve_zoom_position)
        self.setWidget(canvas)
        self.setWidgetResizable(False)
        self.setFrameShape(QScrollArea.NoFrame)
        self.setAlignment(Qt.AlignCenter)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)

    def _preserve_zoom_position(self, old_zoom: float, new_zoom: float, cursor: QPoint) -> None:
        if new_zoom == 1.0:
            self.horizontalScrollBar().setValue(0)
            self.verticalScrollBar().setValue(0)
            return
        if cursor.x() < 0 or cursor.y() < 0:
            cursor = self.viewport().rect().center()
        else:
            cursor = self.canvas.mapTo(self.viewport(), cursor)
        factor = new_zoom / old_zoom
        horizontal_value = self.horizontalScrollBar().value()
        vertical_value = self.verticalScrollBar().value()

        def apply_scroll_position() -> None:
            horizontal = round((horizontal_value + cursor.x()) * factor - cursor.x())
            vertical = round((vertical_value + cursor.y()) * factor - cursor.y())
            self.horizontalScrollBar().setValue(horizontal)
            self.verticalScrollBar().setValue(vertical)

        QTimer.singleShot(0, apply_scroll_position)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self.canvas.set_viewport_size(self.viewport().width(), self.viewport().height())


class PredictionWorker(QThread):
    completed = pyqtSignal(object, object)
    failed = pyqtSignal(str)

    def __init__(self, viewer, sequence, condition, input_selection):
        super().__init__(viewer)
        self.viewer = viewer
        self.sequence = sequence
        self.condition = condition
        self.input_selection = input_selection

    def run(self) -> None:
        try:
            image, overlay = self.viewer._predict_worker(
                self.sequence, self.condition, self.input_selection
            )
            self.completed.emit(image, overlay)
        except Exception as error:  # Keep inference failures inside the GUI.
            self.failed.emit(str(error))


class FullMapPredictionWorker(QThread):
    progress = pyqtSignal(int, int, int, str)
    completed = pyqtSignal(int, str, str)
    failed = pyqtSignal(str)

    def __init__(self, job_id, captures, manifest, dataset_root, checkpoint,
                 output_dir, condition, tile_size, resolution, device,
                 ignore_outliers=False, parent=None):
        super().__init__(parent)
        self.job_id = job_id
        self.captures = captures
        self.manifest = manifest
        self.dataset_root = dataset_root
        self.checkpoint = checkpoint
        self.output_dir = output_dir
        self.condition = condition
        self.tile_size = tile_size
        self.resolution = resolution
        self.device = device
        self.ignore_outliers = ignore_outliers
        self.cancel_requested = threading.Event()

    def cancel(self) -> None:
        self.cancel_requested.set()

    def run(self) -> None:
        try:
            import cv2
            import torch
            from map.merging.merge_full_map import (
                capture_corners,
                feather_mask,
                map_geometry,
                world_to_pixel,
            )

            model = _load_utae_model(self.checkpoint, self.device)
            min_x, min_y, max_x, max_y, width, height = map_geometry(
                self.manifest, self.resolution
            )
            probability_sum = np.zeros((height, width), dtype=np.float32)
            weight_sum = np.zeros((height, width), dtype=np.float32)
            source_mask = feather_mask(self.tile_size, self.tile_size, True)
            capture_groups = {}
            for capture in self.captures:
                key = (capture.get("row"), capture.get("column"))
                capture_groups.setdefault(key, []).append(capture)
            groups = list(capture_groups.values())
            total = len(groups)
            processed = 0
            skipped = 0

            def predict_group(group):
                rgb_sequence = torch.zeros(
                    (1, len(CONDITIONS), 3, self.tile_size, self.tile_size),
                    dtype=torch.float32,
                    device=self.device,
                )
                rgb_available = torch.zeros(
                    (1, len(CONDITIONS)), dtype=torch.bool, device=self.device
                )
                thermal_sequence = torch.zeros_like(rgb_sequence)
                thermal_available = torch.zeros(
                    (1, len(CONDITIONS)), dtype=torch.bool, device=self.device
                )
                for capture in group:
                    capture_condition = f"{capture.get('season')}_{capture.get('daytime')}"
                    if capture_condition not in CONDITIONS:
                        continue
                    condition_index = CONDITIONS.index(capture_condition)
                    image_path = self.dataset_root / capture["image_file"]
                    rgb_sequence[0, condition_index] = _utae_image_tensor(
                        image_path, self.tile_size, self.device
                    )
                    rgb_available[0, condition_index] = True
                    thermal_path = self.dataset_root / "thermal_images" / image_path.name
                    if not thermal_path.exists():
                        continue
                    thermal_sequence[0, condition_index] = _utae_image_tensor(
                        thermal_path, self.tile_size, self.device
                    )
                    thermal_available[0, condition_index] = True
                if not rgb_available.any():
                    raise ValueError("group contains no RGB images")
                conditions = list(CONDITIONS)
                metadata = _utae_metadata(
                    conditions, rgb_available, thermal_available, self.device
                )
                with torch.inference_mode():
                    logits = model(
                        rgb=rgb_sequence,
                        thermal=thermal_sequence,
                        rgb_frame_valid=rgb_available,
                        thermal_frame_valid=thermal_available,
                        **metadata,
                    )
                probability = torch.sigmoid(logits[0, 0]).cpu().numpy().astype(np.float32)
                return probability.copy()

            def positive_pixel_count(probability):
                return int(np.count_nonzero(probability >= 0.5))

            positive_counts = []
            if self.ignore_outliers:
                for index, group in enumerate(groups, start=1):
                    if self.cancel_requested.is_set():
                        return
                    try:
                        positive_counts.append(positive_pixel_count(predict_group(group)))
                    except (KeyError, OSError, TypeError, ValueError) as error:
                        positive_counts.append(None)
                        self.progress.emit(self.job_id, index, total * 2, f"Skipped: {error}")
                        continue
                    self.progress.emit(self.job_id, index, total * 2, "Scoring prediction outliers")
                valid_counts = np.asarray(
                    [count for count in positive_counts if count is not None], dtype=np.float32
                )
                if valid_counts.size:
                    median = float(np.median(valid_counts))
                    mad = float(np.median(np.abs(valid_counts - median)))
                    outlier_threshold = (
                        median + 3.0 * mad
                        if mad > 0
                        else float(np.percentile(valid_counts, 99))
                    )
                else:
                    outlier_threshold = float("inf")
                outlier_indices = {
                    index for index, count in enumerate(positive_counts)
                    if count is not None and count > outlier_threshold
                }
            else:
                outlier_threshold = None
                outlier_indices = set()

            for index, group in enumerate(groups, start=1):
                if self.cancel_requested.is_set():
                    return
                try:
                    if index - 1 in outlier_indices:
                        probability = np.zeros((self.tile_size, self.tile_size), dtype=np.float32)
                        detail = (
                            f"Zeroed outlier at row {group[0].get('row')}, "
                            f"column {group[0].get('column')}"
                        )
                    else:
                        probability = predict_group(group)
                        detail = f"row {group[0].get('row')}, column {group[0].get('column')}"

                    placement_capture = next(
                        (item for item in group
                         if f"{item.get('season')}_{item.get('daytime')}" == self.condition),
                        group[0],
                    )
                    metadata_path = self.dataset_root / placement_capture["metadata_file"]
                    with metadata_path.open() as stream:
                        metadata = json.load(stream)
                    corners = capture_corners(
                        placement_capture, metadata, float(self.manifest["tile_size_m"])
                    )
                    destination = world_to_pixel(corners, min_x, min_y, self.resolution)
                    x0 = max(0, int(np.floor(destination[:, 0].min())) - 1)
                    y0 = max(0, int(np.floor(destination[:, 1].min())) - 1)
                    x1 = min(width, int(np.ceil(destination[:, 0].max())) + 2)
                    y1 = min(height, int(np.ceil(destination[:, 1].max())) + 2)
                    if x1 <= x0 or y1 <= y0:
                        raise ValueError("capture does not intersect map bounds")
                    source = np.float32([
                        [0, 0],
                        [self.tile_size - 1, 0],
                        [self.tile_size - 1, self.tile_size - 1],
                        [0, self.tile_size - 1],
                    ])
                    transform = cv2.getPerspectiveTransform(
                        source, destination - np.array([x0, y0], dtype=np.float32)
                    )
                    size = (x1 - x0, y1 - y0)
                    warped_probability = cv2.warpPerspective(probability, transform, size)
                    warped_weight = cv2.warpPerspective(source_mask, transform, size)
                    valid = warped_weight > 0
                    probability_region = probability_sum[y0:y1, x0:x1]
                    weight_region = weight_sum[y0:y1, x0:x1]
                    probability_region[valid] += (
                        warped_probability[valid] * warped_weight[valid]
                    )
                    weight_region[valid] += warped_weight[valid]
                    processed += 1
                except (KeyError, OSError, TypeError, ValueError) as error:
                    skipped += 1
                    self.progress.emit(
                        self.job_id,
                        total + index if self.ignore_outliers else index,
                        total * 2 if self.ignore_outliers else total,
                        f"Skipped: {error}",
                    )
                    continue
                self.progress.emit(
                    self.job_id, total + index if self.ignore_outliers else index,
                    total * 2 if self.ignore_outliers else total,
                    detail,
                )

            if self.cancel_requested.is_set():
                return
            probability_map = probability_sum / np.maximum(weight_sum, 1e-6)
            probability_map[weight_sum <= 0] = 0
            self.output_dir.mkdir(parents=True, exist_ok=True)
            stem = "full_prediction"
            probability_path = self.output_dir / f"{stem}.npy"
            image_path = self.output_dir / f"{stem}.png"
            metadata_path = self.output_dir / f"{stem}.json"
            temporary_probability = probability_path.with_suffix(".tmp.npy")
            temporary_image = image_path.with_suffix(".tmp.png")
            temporary_metadata = metadata_path.with_suffix(".tmp.json")
            np.save(temporary_probability, probability_map)
            colorized = __import__("matplotlib.cm").cm.get_cmap("inferno")(
                np.clip(probability_map, 0, 1)
            )
            colorized_rgb = (colorized[..., :3] * 255).astype(np.uint8)
            if not cv2.imwrite(
                str(temporary_image), cv2.cvtColor(colorized_rgb, cv2.COLOR_RGB2BGR)
            ):
                raise OSError(f"Could not write {temporary_image}")
            cache_metadata = {
                "condition": "full",
                "source_dataset": str(self.dataset_root),
                "manifest_mtime_ns": self._mtime(self.dataset_root / "map_manifest.json"),
                "checkpoint": str(self.checkpoint),
                "checkpoint_mtime_ns": self._mtime(self.checkpoint),
                "resolution_m_per_pixel": self.resolution,
                "world_bounds_xy": self.manifest["map_bounds_xy"],
                "image_size": [width, height],
                "captures_total": total,
                "captures_processed": processed,
                "captures_skipped": skipped,
                "ignore_outliers": self.ignore_outliers,
                "positive_probability_threshold": 0.5,
                "outlier_method": "median_plus_3_mad" if self.ignore_outliers else None,
                "outlier_threshold_positive_pixels": outlier_threshold,
                "outliers_ignored": len(outlier_indices),
                "orientation": "world_x_down_world_y_right",
                "colormap": "inferno",
                "image_color_order": "rgb",
                "prediction_orientation": "world_aligned_from_raw_input",
                "model_type": "phase2_rgb_thermal_image_utae",
            }
            with temporary_metadata.open("w") as stream:
                json.dump(cache_metadata, stream, indent=2)
            temporary_probability.replace(probability_path)
            temporary_image.replace(image_path)
            temporary_metadata.replace(metadata_path)
            self.completed.emit(self.job_id, str(image_path), str(metadata_path))
        except Exception as error:
            self.failed.emit(str(error))

    @staticmethod
    def _mtime(path: Path) -> int:
        return path.stat().st_mtime_ns


class SettingsDialog(QDialog):
    def __init__(self, parent, dataset_dir, map_dir, checkpoint, map_size, world_x, world_y):
        super().__init__(parent)
        self.setWindowTitle("Map Settings")
        self.setMinimumWidth(620)
        form = QFormLayout(self)

        self.dataset_edit = self._path_row(form, "Dataset folder", dataset_dir, False)
        self.map_edit = self._path_row(form, "Map images folder", map_dir, False)
        self.checkpoint_edit = self._path_row(form, "Fusion model", checkpoint, True)
        self.width_edit = QLineEdit(str(map_size[0] if map_size else ""))
        self.height_edit = QLineEdit(str(map_size[1] if map_size else ""))
        form.addRow("Map width (pixels)", self.width_edit)
        form.addRow("Map height (pixels)", self.height_edit)
        self.x_min_edit, self.x_max_edit = self._pair_rows(form, "World X", world_x)
        self.y_min_edit, self.y_max_edit = self._pair_rows(form, "World Y", world_y)

        buttons = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self._validate_and_accept)
        buttons.rejected.connect(self.reject)
        form.addRow(buttons)

    @staticmethod
    def _path_row(form, label, value, file_mode):
        edit = QLineEdit(str(value))
        browse = QPushButton("Browse")
        if file_mode:
            browse.clicked.connect(lambda: edit.setText(
                QFileDialog.getOpenFileName(
                    form.parentWidget(), label, edit.text(), "Model files (*)"
                )[0] or edit.text()
            ))
        else:
            browse.clicked.connect(lambda: edit.setText(
                QFileDialog.getExistingDirectory(form.parentWidget(), label, edit.text()) or edit.text()
            ))
        row = QHBoxLayout()
        row.addWidget(edit)
        row.addWidget(browse)
        form.addRow(label, row)
        return edit

    @staticmethod
    def _pair_rows(form, label, values):
        row = QHBoxLayout()
        minimum = QLineEdit(str(values[0]))
        maximum = QLineEdit(str(values[1]))
        row.addWidget(QLabel("min"))
        row.addWidget(minimum)
        row.addWidget(QLabel("max"))
        row.addWidget(maximum)
        form.addRow(label, row)
        return minimum, maximum

    def _validate_and_accept(self):
        try:
            dataset = _config_path(self.dataset_edit.text())
            map_images = _config_path(self.map_edit.text())
            checkpoint = _config_path(self.checkpoint_edit.text())
            if not dataset.is_dir():
                raise ValueError(f"Dataset folder does not exist: {dataset}")
            if not map_images.is_dir():
                raise ValueError(f"Map images folder does not exist: {map_images}")
            if not checkpoint.is_file():
                raise ValueError(f"Fusion model does not exist: {checkpoint}")
            if not (dataset / "map_manifest.json").is_file():
                raise ValueError(f"Dataset manifest not found: {dataset / 'map_manifest.json'}")
            map_size = int(self.width_edit.text()), int(self.height_edit.text())
            world_x = float(self.x_min_edit.text()), float(self.x_max_edit.text())
            world_y = float(self.y_min_edit.text()), float(self.y_max_edit.text())
            if min(map_size) <= 0 or world_x[0] >= world_x[1] or world_y[0] >= world_y[1]:
                raise ValueError("Map size must be positive and coordinate minimums must be smaller than maximums")
        except (OSError, TypeError, ValueError) as error:
            QMessageBox.warning(self, "Invalid settings", str(error))
            return
        self._settings = {
            "dataset": dataset,
            "map_images": map_images,
            "checkpoint": checkpoint,
            "map_size": map_size,
            "world_x": world_x,
            "world_y": world_y,
        }
        self.accept()

    def settings(self):
        return self._settings


class InputImagesDialog(QDialog):
    def __init__(self, parent, captures, condition, selected, current_only,
                 read_image, dataset_image_path, thermal_path):
        super().__init__(parent)
        self.setWindowTitle("Model Input Images")
        self.setMinimumSize(760, 620)
        self._checks = {"rgb": {}, "thermal": {}}
        self._captures = captures
        self._read_image = read_image
        self._dataset_image_path = dataset_image_path
        self._thermal_path = thermal_path

        outer = QVBoxLayout(self)
        outer.addWidget(QLabel("Choose the RGB and thermal frames used for this prediction."))
        self.current_only_check = QPushButton("Use only current RGB + thermal")
        self.current_only_check.setCheckable(True)
        self.current_only_check.setChecked(current_only)
        self.current_only_check.toggled.connect(
            lambda checked: self._set_current_only(checked, condition)
        )
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        content = QWidget()
        grid = QGridLayout(content)
        grid.addWidget(QLabel("RGB"), 0, 0, 1, 2)
        grid.addWidget(QLabel("Thermal"), 0, 2, 1, 2)
        for row, capture in enumerate(captures, start=1):
            condition = f"{capture.get('season')}_{capture.get('daytime')}"
            rgb_path = dataset_image_path(capture)
            thermal_image_path = thermal_path(capture)
            self._add_image_option(
                grid, row, 0, "rgb", condition, rgb_path,
                selected.get("rgb", set()),
            )
            self._add_image_option(
                grid, row, 2, "thermal", condition, thermal_image_path,
                selected.get("thermal", set()),
            )
        scroll.setWidget(content)
        outer.addWidget(scroll, 1)

        actions = QHBoxLayout()
        select_all = QPushButton("Select All")
        clear_all = QPushButton("Clear All")
        select_all.clicked.connect(lambda: self._set_all(True))
        clear_all.clicked.connect(lambda: self._set_all(False))
        actions.addWidget(select_all)
        actions.addWidget(clear_all)
        actions.addWidget(self.current_only_check)
        actions.addStretch()
        outer.addLayout(actions)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self._validate_and_accept)
        buttons.rejected.connect(self.reject)
        outer.addWidget(buttons)

    def _add_image_option(self, grid, row, column, modality, condition, path, selected):
        check = QCheckBox(condition)
        check.setChecked(condition in selected and path is not None and path.exists())
        check.setEnabled(path is not None and path.exists())
        preview = QLabel("Unavailable")
        preview.setAlignment(Qt.AlignCenter)
        preview.setFixedSize(145, 100)
        if path is not None and path.exists():
            try:
                image = self._read_image(path).transpose(Image.Transpose.ROTATE_180)
                preview.setPixmap(
                    pil_to_pixmap(image).scaled(
                        preview.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
                    )
                )
            except (OSError, ValueError):
                pass
        self._checks[modality][condition] = check
        grid.addWidget(check, row, column)
        grid.addWidget(preview, row, column + 1)

    def _set_all(self, checked):
        self.current_only_check.setChecked(False)
        for checks in self._checks.values():
            for check in checks.values():
                if check.isEnabled():
                    check.setChecked(checked)

    def _set_current_only(self, checked, condition):
        if checked:
            for checks in self._checks.values():
                for name, check in checks.items():
                    if check.isEnabled():
                        check.setChecked(name == condition)

    def _validate_and_accept(self):
        selected = self.settings()
        if not selected["rgb"] and not selected["thermal"]:
            QMessageBox.warning(self, "No model inputs", "Select at least one RGB or thermal image.")
            return
        self._selected = selected
        self.accept()

    def settings(self):
        selected = {
            modality: {condition for condition, check in checks.items() if check.isChecked()}
            for modality, checks in self._checks.items()
        }
        selected["current_only"] = self.current_only_check.isChecked()
        return selected


class MapViewer(QMainWindow):
    def __init__(self, args: argparse.Namespace):
        super().__init__()
        self.args = args
        self.map_dir = args.map_dir
        self.config_path = args.config
        self.map_size = args.map_size
        self.sequences, world_x, world_y, self.tile_size, self.manifest = load_manifest(args.manifest)
        self.world_x = args.world_x or world_x
        self.world_y = args.world_y or world_y
        self.sequences = [sequence for sequence in self.sequences if sequence_position(sequence)]
        if not self.sequences:
            raise ValueError("The manifest contains no sequences with coordinates")

        available_seasons = self.manifest.get("seasons", [])
        available_daytimes = self.manifest.get("daytimes", [])
        self.season = "summer" if "summer" in available_seasons else SEASONS[0]
        self.daytime = "day" if "day" in available_daytimes else DAYTIMES[0]
        self.prediction_enabled = False
        self.current_only = False
        self.input_selection: dict[str, set[str]] | None = None
        self.selection: dict | None = None
        self.image_cache: dict[Path, Image.Image] = {}
        self.model = None
        self.worker: PredictionWorker | None = None
        self.prediction_map_worker: FullMapPredictionWorker | None = None
        self.prediction_map_job = 0

        self._build_ui()
        self.map_canvas.set_world_bounds(self.world_x, self.world_y)
        self._load_map()
        self._refresh_selection()

    def _open_settings(self) -> None:
        dialog = SettingsDialog(
            self,
            dataset_dir=self.args.manifest.parent,
            map_dir=self.map_dir,
            checkpoint=self.args.checkpoint,
            map_size=self.map_size,
            world_x=self.world_x,
            world_y=self.world_y,
        )
        if dialog.exec_() != QDialog.Accepted:
            return
        settings = dialog.settings()
        try:
            manifest_path = settings["dataset"] / "map_manifest.json"
            load_manifest(manifest_path)
            self.config_path.parent.mkdir(parents=True, exist_ok=True)
            with self.config_path.open("w") as stream:
                def config_value(path):
                    try:
                        return str(path.relative_to(ROOT))
                    except ValueError:
                        return str(path)
                json.dump({
                    "dataset": config_value(settings["dataset"]),
                    "map_images": config_value(settings["map_images"]),
                    "checkpoint": config_value(settings["checkpoint"]),
                    "map_size": list(settings["map_size"]),
                    "world_x": list(settings["world_x"]),
                    "world_y": list(settings["world_y"]),
                }, stream, indent=2)
            self._reload_configuration(
                manifest_path,
                settings["map_images"],
                settings["checkpoint"],
                settings["map_size"],
                settings["world_x"],
                settings["world_y"],
            )
        except (OSError, ValueError) as error:
            QMessageBox.critical(self, "Settings", str(error))

    def _reload_configuration(
        self,
        manifest_path: Path,
        map_dir: Path,
        checkpoint: Path,
        map_size: tuple[int, int],
        world_x: tuple[float, float],
        world_y: tuple[float, float],
    ) -> None:
        sequences, manifest_x, manifest_y, tile_size, manifest = load_manifest(manifest_path)
        sequences = [sequence for sequence in sequences if sequence_position(sequence)]
        if not sequences:
            raise ValueError("The manifest contains no sequences with coordinates")
        self._cancel_prediction_map()
        self.args.manifest = manifest_path
        self.args.map_dir = map_dir
        self.args.checkpoint = checkpoint
        self.map_dir = map_dir
        self.map_size = map_size
        self.sequences = sequences
        self.tile_size = tile_size
        self.manifest = manifest
        self.world_x = world_x or manifest_x
        self.world_y = world_y or manifest_y
        available_seasons = self.manifest.get("seasons", [])
        available_daytimes = self.manifest.get("daytimes", [])
        self.season = self.season if self.season in available_seasons else SEASONS[0]
        self.daytime = self.daytime if self.daytime in available_daytimes else DAYTIMES[0]
        for button, value in zip(self.season_buttons, SEASONS):
            button.blockSignals(True)
            button.setChecked(value == self.season)
            button.blockSignals(False)
        for button, value in zip(self.daytime_buttons, DAYTIMES):
            button.blockSignals(True)
            button.setChecked(value == self.daytime)
            button.blockSignals(False)
        self.map_canvas.set_world_bounds(self.world_x, self.world_y)
        self.image_cache.clear()
        self.model = None
        self.selection = None
        self._load_map()
        self._refresh_selection()

    @property
    def condition(self) -> str:
        return f"{self.season}_{self.daytime}"

    def _build_ui(self) -> None:
        self.setWindowTitle("RGB / Thermal Map Viewer")
        self.setFixedSize(1750, 1000)
        root = QWidget()
        self.setCentralWidget(root)
        outer = QVBoxLayout(root)
        outer.setContentsMargins(12, 12, 12, 12)
        outer.setSpacing(10)

        controls = QWidget(objectName="toolbar")
        control_layout = QHBoxLayout(controls)
        control_layout.setContentsMargins(12, 7, 12, 7)
        control_layout.setSpacing(5)

        season_box = QHBoxLayout()
        season_box.addWidget(QLabel("SEASON", objectName="controlLabel"))
        season_group = QButtonGroup(self)
        season_group.setExclusive(True)
        self.season_buttons = []
        for season in SEASONS:
            button = QRadioButton(season.title())
            button.setChecked(season == self.season)
            button.clicked.connect(lambda _checked, value=season: self._set_season(value))
            season_group.addButton(button)
            season_box.addWidget(button)
            self.season_buttons.append(button)
        control_layout.addLayout(season_box)

        daytime_box = QHBoxLayout()
        daytime_box.setContentsMargins(18, 0, 0, 0)
        daytime_box.addWidget(QLabel("DAYTIME", objectName="controlLabel"))
        daytime_group = QButtonGroup(self)
        daytime_group.setExclusive(True)
        self.daytime_buttons = []
        for daytime in DAYTIMES:
            button = QRadioButton(daytime.title())
            button.setChecked(daytime == self.daytime)
            button.clicked.connect(lambda _checked, value=daytime: self._set_daytime(value))
            daytime_group.addButton(button)
            daytime_box.addWidget(button)
            self.daytime_buttons.append(button)
        control_layout.addLayout(daytime_box)
        self.prediction_progress = QProgressBar()
        self.prediction_progress.setRange(0, 100)
        self.prediction_progress.setValue(0)
        self.prediction_progress.setFixedWidth(180)
        self.prediction_progress.setFixedHeight(26)
        self.prediction_progress.setFormat("Prediction map: %p%")
        self.prediction_progress.setVisible(False)
        control_layout.addStretch()
        control_layout.addWidget(self.prediction_progress)
        control_layout.addStretch()

        # zoom_box = QHBoxLayout()
        # zoom_box.addWidget(QLabel("ZOOM", objectName="controlLabel"))
        # zoom_out = QPushButton("-")
        # zoom_in = QPushButton("+")
        # reset = QPushButton("Reset")
        # zoom_out.clicked.connect(lambda: self.map_canvas.zoom_by(0.8))
        # zoom_in.clicked.connect(lambda: self.map_canvas.zoom_by(1.25))
        # reset.clicked.connect(lambda: self.map_canvas.reset_zoom())
        # for button in (zoom_out, zoom_in, reset):
        #     button.setObjectName("zoomButton")
        #     zoom_box.addWidget(button)
        # control_layout.addLayout(zoom_box)

        self.input_images_button = QPushButton("Input Images")
        self.input_images_button.setEnabled(False)
        self.input_images_button.clicked.connect(self._open_input_images)
        self.prediction_toggle = QCheckBox("Show landmine prediction")
        self.prediction_toggle.stateChanged.connect(self._prediction_mode_changed)
        control_layout.addWidget(self.prediction_toggle)
        self.prediction_overlay_toggle = QCheckBox("Overlay prediction map")
        self.prediction_overlay_toggle.stateChanged.connect(self._prediction_overlay_changed)
        control_layout.addWidget(self.prediction_overlay_toggle)
        self.outlier_toggle = QCheckBox("Ignore outlier prediction images")
        self.outlier_toggle.stateChanged.connect(self._outlier_mode_changed)
        self.prediction_opacity_slider = QSlider(Qt.Horizontal)
        self.prediction_opacity_slider.setRange(0, 100)
        self.prediction_opacity_slider.setValue(52)
        self.prediction_opacity_slider.setFixedWidth(85)
        self.prediction_opacity_slider.setFixedHeight(10)
        self.prediction_opacity_slider.setEnabled(False)
        self.prediction_opacity_slider.valueChanged.connect(self._prediction_opacity_changed)
        control_layout.addWidget(self.prediction_opacity_slider)
        self.prediction_opacity_value = QLabel("52%", objectName="controlLabel")
        self.prediction_opacity_value.setFixedWidth(32)
        control_layout.addWidget(self.prediction_opacity_value)
        control_layout.addWidget(self.input_images_button)
        settings_button = QPushButton("Settings")
        settings_button.clicked.connect(self._open_settings)
        control_layout.addWidget(settings_button)
        outer.addWidget(controls)

        content = QHBoxLayout()
        content.setSpacing(10)
        map_group = QGroupBox("OVERALL MAP")
        map_group.setFixedWidth(930)
        map_layout = QVBoxLayout(map_group)
        map_layout.setContentsMargins(7, 7, 7, 7)
        self.map_canvas = MapCanvas()
        self.map_canvas.map_clicked.connect(self._map_clicked)
        self.map_scroll_area = MapScrollArea(self.map_canvas)
        map_layout.addWidget(self.map_scroll_area)
        content.addWidget(map_group)

        detail_group = QGroupBox("SELECTED POINT")
        detail_group.setFixedWidth(770)
        detail_layout = QVBoxLayout(detail_group)
        detail_layout.setContentsMargins(7, 7, 7, 7)
        self.status_label = QLabel()
        self.status_label.setObjectName("statusLabel")
        self.status_label.setWordWrap(True)
        point_controls = QHBoxLayout()
        point_controls.addWidget(self.status_label, 1)
        detail_layout.addLayout(point_controls)

        image_grid = QHBoxLayout()
        image_grid.setSpacing(8)
        left = QVBoxLayout()
        right = QVBoxLayout()
        self.rgb_panel = self._image_panel("RGB IMAGE", left)
        self.prediction_panel = self._image_panel("FUSION PREDICTION", left)
        self.thermal_panel = self._image_panel("THERMAL IMAGE", right)
        self.overlay_panel = self._image_panel("PREDICTION OVERLAY (RGB)", right)
        image_grid.addLayout(left)
        image_grid.addLayout(right)
        detail_layout.addLayout(image_grid, 1)
        content.addWidget(detail_group)
        outer.addLayout(content, 1)

        self.setStyleSheet(
            """
             QMainWindow, QWidget { background: #171717; color: #f5f5f4; font-family: Ubuntu; }
             QWidget#toolbar { background: #1f1f1f; border: 1px solid #404040; border-radius: 10px; }
            QLabel#controlLabel { background: transparent; color: #a8a29e; font-size: 9pt; font-weight: 700; }
            QGroupBox { background: #242424; border: 1px solid #404040; border-radius: 10px; margin-top: 12px; padding-top: 8px; font-size: 10pt; font-weight: 700; }
            QGroupBox::title { subcontrol-origin: margin; left: 14px; padding: 0 5px; color: #f5f5f4; }
            QRadioButton, QCheckBox { background: transparent; color: #a8a29e; spacing: 6px; padding: 3px; font-size: 9pt; }
            QRadioButton:checked, QCheckBox:checked { color: #f59e0b; }
            QRadioButton::indicator, QCheckBox::indicator { width: 16px; height: 16px; border-radius: 8px; border: 2px solid #78716c; background: #292524; }
            QRadioButton::indicator:checked, QCheckBox::indicator:checked { border-color: #f59e0b; background: #f59e0b; }
            QPushButton { background: #292524; color: #f5f5f4; border: 1px solid #404040; border-radius: 7px; padding: 6px 11px; font-weight: 700; }
            QPushButton:hover { background: #44403c; border-color: #78716c; }
             QPushButton:pressed { background: #b45309; }
             QPushButton#zoomButton { min-width: 28px; padding: 5px 8px; }
             QProgressBar { background: #292524; color: #f5f5f4; border: 1px solid #404040; border-radius: 7px; padding: 1px; text-align: center; }
             QProgressBar::chunk { background: #f59e0b; margin: 1px; border-radius: 5px; }
             QSlider::groove:horizontal { height: 6px; background: #57534e; border-radius: 1px; }
             QSlider::sub-page:horizontal { background: #f59e0b; border-radius: 2px; }
             QSlider::add-page:horizontal { background: #292524; border-radius: 2px; }
             QSlider::handle:horizontal { width: 10px; height: 10px; margin: -2px 0; background: #a8a29e; border: 1px solid #d6d3d1; border-radius: 4px; }
             QSlider::handle:horizontal:hover { background: #d6d3d1; }
             QSlider::groove:horizontal:disabled { background: #3f3f46; }
             QSlider::sub-page:horizontal:disabled { background: #57534e; }
             QSlider::handle:horizontal:disabled { background: #57534e; border-color: #78716c; }
             QScrollArea { background: #171717; border: 1px solid #404040; border-radius: 7px; }
             QScrollBar:horizontal, QScrollBar:vertical { background: #292524; border: 1px solid #404040; margin: 2px; }
             QScrollBar:horizontal { height: 13px; }
             QScrollBar:vertical { width: 13px; }
             QScrollBar::handle:horizontal, QScrollBar::handle:vertical { background: #78716c; border-radius: 5px; min-width: 28px; min-height: 28px; }
             QScrollBar::handle:hover { background: #f59e0b; }
             QScrollBar::add-line, QScrollBar::sub-line { background: #44403c; border: none; }
             QScrollBar::add-page, QScrollBar::sub-page { background: #1f1f1f; }
             QLabel#statusLabel { background: #292524; color: #f5f5f4; border-radius: 8px; padding: 10px 12px; font-size: 10pt; }
            QLabel#imagePanel { background: #1f1f1f; color: #a8a29e; border: 1px solid #404040; border-radius: 8px; }
            """
        )

    @staticmethod
    def _image_panel(title: str, layout: QVBoxLayout) -> QLabel:
        group = QGroupBox(title)
        group.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        group_layout = QVBoxLayout(group)
        group_layout.setContentsMargins(5, 5, 5, 5)
        label = QLabel("No point selected")
        label.setObjectName("imagePanel")
        label.setAlignment(Qt.AlignCenter)
        label.setMinimumSize(300, 260)
        label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        group_layout.addWidget(label)
        layout.addWidget(group, 1)
        return label

    def _set_season(self, value: str) -> None:
        self.season = value
        self._condition_changed()

    def _set_daytime(self, value: str) -> None:
        self.daytime = value
        self._condition_changed()

    def _input_captures(self) -> list[dict]:
        if self.selection is None:
            return []
        return [
            capture for capture in self.sequences
            if capture.get("row") == self.selection.get("row")
            and capture.get("column") == self.selection.get("column")
        ]

    def _effective_input_selection(self) -> dict[str, set[str]]:
        captures = self._input_captures()
        available = {
            "rgb": {
                f"{capture.get('season')}_{capture.get('daytime')}"
                for capture in captures
                if self._dataset_image_path(capture) is not None
                and self._dataset_image_path(capture).exists()
            },
            "thermal": {
                f"{capture.get('season')}_{capture.get('daytime')}"
                for capture in captures
                if self._dataset_thermal_path(capture) is not None
                and self._dataset_thermal_path(capture).exists()
            },
        }
        if self.current_only:
            current = {self.condition}
            return {modality: values & current for modality, values in available.items()}
        if self.input_selection is None:
            return available
        return {
            modality: values & self.input_selection.get(modality, set())
            for modality, values in available.items()
        }

    def _open_input_images(self) -> None:
        captures = self._input_captures()
        if not captures:
            return
        selected = self._effective_input_selection()
        dialog = InputImagesDialog(
            self,
            captures,
            self.condition,
            selected,
            self.current_only,
            self._read_image,
            self._dataset_image_path,
            self._dataset_thermal_path,
        )
        if dialog.exec_() != QDialog.Accepted:
            return
        settings = dialog.settings()
        self.input_selection = {
            "rgb": settings["rgb"],
            "thermal": settings["thermal"],
        }
        self.current_only = settings["current_only"]
        if self.prediction_enabled:
            self._start_prediction()

    def _load_map(self) -> None:
        path = self.map_dir / f"{self.condition}.png"
        if not path.exists():
            self.map_canvas.clear_map()
            return False
        with Image.open(path) as source:
            if self.map_size is None:
                self.map_size = source.size
            self.map_canvas.set_map_image(source.convert("RGB"), self.map_size)
        self.map_canvas.set_prediction_overlay(None)
        self.map_canvas.set_map_orientation(False)
        return True

    def _condition_changed(self) -> None:
        self._cancel_prediction_map()
        self.prediction_overlay_toggle.blockSignals(True)
        self.prediction_overlay_toggle.setChecked(False)
        self.prediction_overlay_toggle.blockSignals(False)
        self.prediction_opacity_slider.setEnabled(False)
        matching = [
            capture for capture in self.sequences
            if capture.get("season") == self.season
            and capture.get("daytime") == self.daytime
        ]
        # Keep the empty state empty when changing conditions.
        if self.selection is not None:
            position = sequence_position(self.selection)
            grid_match = [
                capture for capture in matching
                if capture.get("row") == self.selection.get("row")
                and capture.get("column") == self.selection.get("column")
            ]
            self.selection = (
                grid_match[0]
                if grid_match
                else min(
                    matching,
                    key=lambda capture: self._distance_squared(capture, *position),
                )
                if matching and position is not None
                else matching[0] if matching else None
            )
        loaded = self._load_map()
        if not loaded:
            self.status_label.setText(f"Condition unavailable: {self.condition}")
        self._refresh_selection()

    def _prediction_cache_paths(self) -> tuple[Path, Path, Path]:
        stem = "full_prediction"
        return (
            self.map_dir / f"{stem}.npy",
            self.map_dir / f"{stem}.png",
            self.map_dir / f"{stem}.json",
        )

    def _prediction_resolution(self) -> float:
        map_metadata = self.map_dir / f"{self.condition}.json"
        if map_metadata.exists():
            try:
                with map_metadata.open() as stream:
                    return float(json.load(stream)["resolution_m_per_pixel"])
            except (KeyError, OSError, TypeError, ValueError):
                pass
        return 0.02

    def _prediction_cache_valid(self) -> bool:
        probability_path, image_path, metadata_path = self._prediction_cache_paths()
        if not all(path.exists() for path in (probability_path, image_path, metadata_path)):
            return False
        try:
            with metadata_path.open() as stream:
                metadata = json.load(stream)
            locations = {(capture.get("row"), capture.get("column")) for capture in self.sequences}
            return (
                metadata.get("condition") == "full"
                and metadata.get("source_dataset") == str(self.args.manifest.parent)
                and metadata.get("manifest_mtime_ns") == self.args.manifest.stat().st_mtime_ns
                and metadata.get("checkpoint") == str(self.args.checkpoint)
                and metadata.get("checkpoint_mtime_ns") == self.args.checkpoint.stat().st_mtime_ns
                and metadata.get("resolution_m_per_pixel") == self._prediction_resolution()
                and metadata.get("world_bounds_xy") == self.manifest["map_bounds_xy"]
                and metadata.get("captures_total") == len(locations)
                and metadata.get("captures_skipped") == 0
                and metadata.get("ignore_outliers") == self.outlier_toggle.isChecked()
                and metadata.get("positive_probability_threshold") == 0.5
                and metadata.get("outlier_method") == (
                    "median_plus_3_mad" if self.outlier_toggle.isChecked() else None
                )
                and metadata.get("prediction_orientation") == "world_aligned_from_raw_input"
                and metadata.get("model_type") == "phase2_rgb_thermal_image_utae"
            )
        except (OSError, TypeError, ValueError):
            return False

    def _prediction_image(self) -> Image.Image:
        _probability_path, image_path, metadata_path = self._prediction_cache_paths()
        with Image.open(image_path) as source:
            image = source.convert("RGB")
        try:
            with metadata_path.open() as stream:
                metadata = json.load(stream)
        except (OSError, TypeError, ValueError):
            metadata = {}
        if metadata.get("image_color_order") != "rgb":
            image = Image.fromarray(np.asarray(image, dtype=np.uint8)[:, :, ::-1].copy())
        return image

    def _load_prediction_overlay(self) -> None:
        self._load_map()
        self.map_canvas.set_prediction_overlay(self._prediction_image())
        self.map_canvas.set_map_orientation(False)
        self._refresh_selection()

    def _prediction_opacity_changed(self, value: int) -> None:
        self.prediction_opacity_value.setText(f"{value}%")
        self.map_canvas.set_prediction_overlay_opacity(value)

    def _prediction_overlay_changed(self, state: int) -> None:
        self.prediction_opacity_slider.setEnabled(bool(state))
        if not state:
            self.map_canvas.set_prediction_overlay(None)
            self._load_map()
            return
        captures = self.sequences
        if not captures:
            self.prediction_overlay_toggle.blockSignals(True)
            self.prediction_overlay_toggle.setChecked(False)
            self.prediction_overlay_toggle.blockSignals(False)
            self.prediction_opacity_slider.setEnabled(False)
            self.status_label.setText(f"Prediction map unavailable for {self.condition}")
            return
        if self._prediction_cache_valid():
            self._load_prediction_overlay()
        else:
            self._start_prediction_map()

    def _outlier_mode_changed(self, _state: int) -> None:
        if self.prediction_overlay_toggle.isChecked():
            self._start_prediction_map()

    def _start_prediction_map(self) -> None:
        self._cancel_prediction_map()
        self.prediction_map_job += 1
        job_id = self.prediction_map_job
        captures = self.sequences
        self.prediction_progress.setValue(0)
        self.prediction_progress.setFormat("Calculating map: 0%")
        self.prediction_progress.setVisible(True)
        self.prediction_map_worker = FullMapPredictionWorker(
            job_id,
            captures,
            self.manifest,
            self.args.manifest.parent,
            self.args.checkpoint,
            self.map_dir,
            self.condition,
            self.args.tile_size,
            self._prediction_resolution(),
            self.args.device,
            self.outlier_toggle.isChecked(),
            self,
        )
        self.prediction_map_worker.progress.connect(self._prediction_map_progress)
        self.prediction_map_worker.completed.connect(self._prediction_map_completed)
        self.prediction_map_worker.failed.connect(self._prediction_map_failed)
        self.prediction_map_worker.start()

    def _prediction_map_progress(self, job_id: int, completed: int, total: int, detail: str) -> None:
        if job_id != self.prediction_map_job:
            return
        percent = round(completed / max(total, 1) * 100)
        # Avoid a tiny square-looking chunk before it can render as a rounded fill.
        self.prediction_progress.setValue(percent if percent >= 7 else 0)
        self.prediction_progress.setFormat(f"Prediction map: {completed}/{total}")

    def _prediction_map_completed(self, job_id: int, _image_path: str, _metadata_path: str) -> None:
        if job_id != self.prediction_map_job:
            return
        self.prediction_progress.setValue(100)
        self.prediction_progress.setVisible(False)
        self.prediction_map_worker = None
        if self.prediction_overlay_toggle.isChecked():
            self._load_prediction_overlay()

    def _prediction_map_failed(self, error: str) -> None:
        if self.prediction_map_worker is not None:
            self.prediction_map_worker = None
        self.prediction_progress.setVisible(False)
        self.prediction_overlay_toggle.blockSignals(True)
        self.prediction_overlay_toggle.setChecked(False)
        self.prediction_overlay_toggle.blockSignals(False)
        self._load_map()
        self._refresh_selection()
        self.status_label.setText(f"Prediction map failed:\n{error}")

    def _cancel_prediction_map(self) -> None:
        self.prediction_map_job += 1
        worker = self.prediction_map_worker
        if worker is not None:
            worker.cancel()
            self.prediction_map_worker = None

    def closeEvent(self, event) -> None:
        prediction_worker = self.prediction_map_worker
        self._cancel_prediction_map()
        if prediction_worker is not None:
            prediction_worker.wait(2000)
        if self.worker is not None and self.worker.isRunning():
            self.worker.wait(2000)
        super().closeEvent(event)

    def _map_clicked(self, x: float, y: float) -> None:
        condition_captures = [
            sequence for sequence in self.sequences
            if sequence.get("season") == self.season
            and sequence.get("daytime") == self.daytime
        ]
        if not condition_captures:
            self.selection = None
            self._refresh_selection()
            return
        containing = [
            sequence for sequence in condition_captures
            if self._capture_contains(sequence, x, y)
        ]
        candidates = containing or condition_captures
        self.selection = min(candidates, key=lambda sequence: self._distance_squared(sequence, x, y))
        self._refresh_selection()

    @staticmethod
    def _capture_contains(capture: dict, x: float, y: float) -> bool:
        corners = capture_corners(capture)
        if corners is None:
            return False
        signs = []
        for index, (x1, y1) in enumerate(corners):
            x2, y2 = corners[(index + 1) % len(corners)]
            signs.append((x2 - x1) * (y - y1) - (y2 - y1) * (x - x1))
        return all(sign >= -1e-6 for sign in signs) or all(sign <= 1e-6 for sign in signs)

    @staticmethod
    def _distance_squared(sequence: dict, x: float, y: float) -> float:
        position = sequence_position(sequence)
        assert position is not None
        return (position[0] - x) ** 2 + (position[1] - y) ** 2

    def _refresh_selection(self) -> None:
        self.map_canvas.set_selected_corners(
            capture_corners(self.selection, self.tile_size) if self.selection else None
        )
        if self.selection is None:
            self.input_images_button.setEnabled(False)
            self.status_label.setText("Click the map to select the closest point")
            self.rgb_panel.setText("No point selected")
            self.thermal_panel.setText("No point selected")
            self._clear_prediction()
            return
        self.input_images_button.setEnabled(bool(self._input_captures()))
        position = sequence_position(self.selection)
        assert position is not None
        self.status_label.setText(
            f"Capture: {self.selection.get('capture_index', 'unknown')}\n"
            f"World position: x={position[0]:.2f}, y={position[1]:.2f}\n"
            f"Condition: {self.condition}"
        )
        rgb_path = self._dataset_image_path(self.selection)
        thermal_path = self._dataset_thermal_path(self.selection)
        self._set_image_label(
            self.rgb_panel,
            rgb_path,
            f"RGB unavailable for {self.condition}",
            rotate_180=True,
        )
        self._set_image_label(
            self.thermal_panel,
            thermal_path,
            "Thermal image unavailable",
            rotate_180=True,
        )
        if (
            (rgb_path is None or not rgb_path.exists())
            and (thermal_path is None or not thermal_path.exists())
        ):
            self._clear_prediction()
            return
        if self.prediction_enabled:
            self._start_prediction()
        else:
            self._clear_prediction()

    @staticmethod
    def _frame_for_condition(sequence: dict, condition: str) -> dict:
        return next((frame for frame in sequence.get("frames", []) if frame.get("condition") == condition), {})

    @staticmethod
    def _frame_path(frame: dict, key: str) -> Path | None:
        value = frame.get(key)
        return Path(value) if value else None

    def _dataset_image_path(self, capture: dict) -> Path | None:
        value = capture.get("image_file")
        return self.args.manifest.parent / value if value else None

    def _dataset_thermal_path(self, capture: dict) -> Path | None:
        image_path = self._dataset_image_path(capture)
        return (
            self.args.manifest.parent / "thermal_images" / image_path.name
            if image_path is not None
            else None
        )

    def _set_image_label(
        self,
        label: QLabel,
        path: Path | None,
        fallback: str,
        rotate_180: bool = False,
    ) -> None:
        if path is None or not path.exists():
            label.clear()
            label.setText(fallback)
            label.setProperty("image_pixmap", False)
            return
        image = self._read_image(path)
        if rotate_180:
            image = image.transpose(Image.Transpose.ROTATE_180)
        label.setPixmap(pil_to_pixmap(image).scaled(label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        label.setProperty("image_pixmap", True)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        if self.selection:
            self._refresh_selection()

    def _read_image(self, path: Path) -> Image.Image:
        if path not in self.image_cache:
            with Image.open(path) as source:
                self.image_cache[path] = source.convert("RGB")
        return self.image_cache[path]

    def _prediction_mode_changed(self, state: int) -> None:
        self.prediction_enabled = bool(state)
        if not self.prediction_enabled:
            self._clear_prediction()
        elif self.selection is None:
            self.prediction_toggle.blockSignals(True)
            self.prediction_toggle.setChecked(False)
            self.prediction_toggle.blockSignals(False)
            self.prediction_enabled = False
            self.status_label.setText("Select a point on the map before enabling prediction")
        else:
            self._start_prediction()

    def _clear_prediction(self) -> None:
        self.prediction_panel.clear()
        self.prediction_panel.setText("Prediction disabled")
        self.overlay_panel.clear()
        self.overlay_panel.setText("Prediction disabled")

    def _start_prediction(self) -> None:
        if self.selection is None:
            return
        if self.worker is not None and self.worker.isRunning():
            return
        self.prediction_panel.setText("Running fusion model...")
        self.overlay_panel.setText("Running fusion model...")
        self.worker = PredictionWorker(
            self, self.selection, self.condition, self._effective_input_selection()
        )
        self.worker.completed.connect(self._show_prediction)
        self.worker.failed.connect(self._prediction_failed)
        self.worker.start()

    def _predict_worker(self, sequence: dict, condition: str, input_selection):
        import torch
        from phase2_rgb_thermal_image_fusion.data import ImageFusionSequenceDataset

        if self.model is None:
            self.model = _load_utae_model(self.args.checkpoint, self.args.device)
        if "image_file" in sequence:
            captures = [
                capture for capture in self.sequences
                if capture.get("row") == sequence.get("row")
                and capture.get("column") == sequence.get("column")
            ]
            if not captures:
                captures = [sequence]
            rgb_sequence = torch.zeros(
                (1, len(CONDITIONS), 3, self.args.tile_size, self.args.tile_size),
                dtype=torch.float32,
                device=self.args.device,
            )
            rgb_available = torch.zeros(
                (1, len(CONDITIONS)), dtype=torch.bool, device=self.args.device
            )
            thermal_sequence = torch.zeros_like(rgb_sequence)
            thermal_available = torch.zeros(
                (1, len(CONDITIONS)), dtype=torch.bool, device=self.args.device
            )
            for capture in captures:
                capture_condition = f"{capture.get('season')}_{capture.get('daytime')}"
                if capture_condition not in CONDITIONS:
                    continue
                condition_index = CONDITIONS.index(capture_condition)
                image_path = self._dataset_image_path(capture)
                if (
                    capture_condition in input_selection.get("rgb", set())
                    and image_path is not None
                    and image_path.exists()
                ):
                    rgb_sequence[0, condition_index] = _utae_image_tensor(
                        image_path, self.args.tile_size, self.args.device
                    )
                    rgb_available[0, condition_index] = True
                thermal_path = self._dataset_thermal_path(capture)
                if (
                    capture_condition in input_selection.get("thermal", set())
                    and thermal_path is not None
                    and thermal_path.exists()
                ):
                    thermal_sequence[0, condition_index] = _utae_image_tensor(
                        thermal_path, self.args.tile_size, self.args.device
                    )
                    thermal_available[0, condition_index] = True
            if not (rgb_available.any() or thermal_available.any()):
                raise FileNotFoundError("No selected RGB or thermal images found for the selected location")
            metadata = _utae_metadata(
                list(CONDITIONS), rgb_available, thermal_available, self.args.device
            )
            with torch.inference_mode():
                logits = self.model(
                    rgb=rgb_sequence,
                    thermal=thermal_sequence,
                    rgb_frame_valid=rgb_available,
                    thermal_frame_valid=thermal_available,
                    **metadata,
                )
            probability = torch.sigmoid(logits[0, 0]).cpu().numpy()
            prediction_image = self._prediction_display_image(probability, sequence)
            return prediction_image, self._prediction_overlay(sequence, probability, condition)
        dataset = ImageFusionSequenceDataset([sequence], tile_size=self.args.tile_size)
        item = dataset[0]
        batch = {
            key: value.unsqueeze(0).to(self.args.device)
            for key, value in item.items()
            if torch.is_tensor(value)
        }
        for condition_index, frame_condition in enumerate(conditions := [
            f"{SEASONS[int(season)]}_{DAYTIMES[int(daytime)]}"
            for season, daytime in zip(
                batch["season_ids"][0].cpu(), batch["daytime_ids"][0].cpu()
            )
        ]):
            if frame_condition in input_selection.get("rgb", set()):
                continue
            batch["rgb_available"][:, condition_index] = False
            batch["rgb_sequence"][:, condition_index] = 0
        for condition_index, frame_condition in enumerate(conditions):
            if frame_condition in input_selection.get("thermal", set()):
                continue
            batch["thermal_available"][:, condition_index] = False
            batch["thermal_image"][:, condition_index] = 0
        metadata = _utae_metadata(
            conditions,
            batch["rgb_available"],
            batch["thermal_available"],
            self.args.device,
        )
        with torch.inference_mode():
            logits = self.model(
                rgb=batch["rgb_sequence"],
                thermal=batch["thermal_image"],
                rgb_frame_valid=batch["rgb_available"],
                thermal_frame_valid=batch["thermal_available"],
                **metadata,
            )
        probability = torch.sigmoid(logits[0, 0]).cpu().numpy()
        prediction_image = self._prediction_display_image(probability, sequence)
        return prediction_image, self._prediction_overlay(sequence, probability, condition)

    def _prediction_display_image(self, probability: np.ndarray, sequence: dict) -> Image.Image:
        if "image_file" in sequence:
            probability = np.rot90(probability, 2).copy()
        rgba = __import__("matplotlib.cm").cm.get_cmap("inferno")(
            np.clip(probability, 0, 1)
        )
        return Image.fromarray((rgba[..., :3] * 255).astype(np.uint8)).resize(
            (self.args.display_size, self.args.display_size), Image.Resampling.BILINEAR
        )

    def _prediction_overlay(self, sequence: dict, probability: np.ndarray, condition: str) -> Image.Image:
        if "image_file" in sequence:
            rgb_path = self._dataset_image_path(sequence)
            if rgb_path is None or not rgb_path.exists():
                return Image.fromarray(np.zeros((self.args.display_size, self.args.display_size, 3), dtype=np.uint8))
            base = np.asarray(
                self._read_image(rgb_path).transpose(Image.Transpose.ROTATE_180),
                dtype=np.float32,
            )
            probability = np.rot90(probability, 2).copy()
        else:
            frame = self._frame_for_condition(sequence, condition)
            rgb_path = self._frame_path(frame, "rgb_path")
            if rgb_path is None or not rgb_path.exists():
                return Image.fromarray(np.zeros((self.args.display_size, self.args.display_size, 3), dtype=np.uint8))
            base = np.asarray(self._read_image(rgb_path), dtype=np.float32)
        resized = np.asarray(
            Image.fromarray((np.clip(probability, 0, 1) * 255).astype(np.uint8)).resize(
                (base.shape[1], base.shape[0]), Image.Resampling.BILINEAR
            ),
            dtype=np.float32,
        ) / 255.0
        mask = resized >= 0.5
        base[mask] = base[mask] * 0.55 + np.asarray((255, 45, 45), dtype=np.float32) * 0.45
        return Image.fromarray(np.clip(base, 0, 255).astype(np.uint8))

    def _show_prediction(self, image: Image.Image, overlay: Image.Image) -> None:
        self.prediction_panel.setPixmap(pil_to_pixmap(image).scaled(self.prediction_panel.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        self.overlay_panel.setPixmap(pil_to_pixmap(overlay).scaled(self.overlay_panel.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def _prediction_failed(self, error: str) -> None:
        self.prediction_panel.setText(f"Prediction failed:\n{error}")
        self.overlay_panel.setText("Prediction unavailable")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_DATASET / "map_manifest.json")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "phase2_rgb_thermal_image_utae/artifacts/run_trees_garbage_fusion_utae/epochs/epoch_0032/checkpoint.pt")
    parser.add_argument("--map-dir", type=Path, default=DEFAULT_MAP_DIR)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--tile-size", type=int, default=640)
    parser.add_argument("--display-size", type=int, default=420)
    parser.add_argument("--device", default="cuda" if _cuda_available() else "cpu")
    return parser.parse_args()


def _cuda_available() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except ImportError:
        return False


def main() -> None:
    args = parse_args()
    config = load_interface_config(args.config)
    dataset = _config_path_value(config, "dataset", args.manifest.parent)
    args.manifest = dataset / "map_manifest.json"
    args.map_dir = _config_path_value(config, "map_images", args.map_dir)
    args.checkpoint = _config_path_value(config, "checkpoint", args.checkpoint)
    args.map_size = _config_size(config)
    args.world_x = _config_pair(config, "world_x")
    args.world_y = _config_pair(config, "world_y")
    for path in (args.manifest, args.map_dir):
        if not path.exists():
            raise SystemExit(f"Required path does not exist: {path}")
    if args.tile_size <= 0 or args.display_size <= 0:
        raise SystemExit("tile size and display size must be positive")
    app = QApplication(sys.argv)
    app.setFont(QFont("Ubuntu", 10))
    try:
        viewer = MapViewer(args)
        viewer.show()
        sys.exit(app.exec_())
    except Exception as error:
        raise SystemExit(str(error)) from error


if __name__ == "__main__":
    main()
