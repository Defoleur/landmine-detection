from omni.isaac.sensor import Camera
import omni.replicator.core as rep
import numpy as np
import cv2
import socket
import threading
import carb
from .dataset_recorder import MINE_LABEL_NAMES

MAX_DGRAM = 65000


def calculate_radiance(absoluteTemperature, emissivity, dx=0.01):
    wavelength = np.arange(8, 14, dx)
    c1 = 1.19104e8
    c2 = 1.43879e4

    radiance = emissivity * (c1 / ((wavelength**5) * (np.exp(c2 / (wavelength * absoluteTemperature)) - 1)))
    return np.trapz(radiance, dx=dx)


# --- Thermal rendering parameters (tune these) ---
THERMAL_PARAMS = {
    "sun_azimuth_deg": 135.0,        # Roughly matches the scene sun (adjust to your lighting).
    "sun_elevation_deg": 45.0,
    "air_temp_K": 288.0,             # Ambient air temperature used for distance haze.
    "air_emissivity": 1.0,
    "atmo_attenuation": 0.012,       # 1/m depth fade rate (larger = more haze).
    "shadow_floor": 0.55,            # Minimum multiplier in cast shadows (1.0 = no darkening).
    "shading_floor": 0.75,           # Minimum Lambertian multiplier for grazing surfaces.
    "shadow_lowpass_sigma": 12.0,    # px blur on luminance before extracting shadows.
    "shadow_norm_percentile": 99.5,  # Normalize luminance by this percentile (robust to hot pixels).
    "texture_amp": 0.0,              # Amplitude of the correlated thermal texture (0 disables).
    "texture_res": 32,               # Low-res noise grid resolution (upsampled).
    "speckle_fraction": 0.0,         # Fraction of hot pixels (0 disables).
    "speckle_gain": 0.0,             # Hot spot intensity (fraction of dynamic range).
    "agc_percentiles": [2.0, 98.0],  # Per-frame stretch range.
    "agc_alpha": 0.05,               # Temporal smoothing (0.0 = fixed, 1.0 = no smoothing). Prevents flicker.
    "readout_noise_sigma": 0.0,      # Per-frame readout noise (DN) (0 disables).
    "fixed_pattern_sigma": 0.0,      # Permanent fixed-pattern bias (DN) (0 disables).
    "dead_pixel_fraction": 0.0,      # Fraction of dead (black) pixels (0 disables).
    "sensor_blur_sigma": 0.7,        # Low-MTF optical blur (0 disables).
}

class ThermalCamera:
    def __init__(self, camera, ip="127.0.0.1", port=5020, resolution=(800, 600), parent_path="/World/quadrotor/body"):
        self.ip = ip
        self.port = port
        self.width = getattr(camera, "width", resolution[0])
        self.height = getattr(camera, "height", resolution[1])

        # --- 1. Create the Isaac Sensor Camera ---
        self.camera = camera._cam

        # --- 2. Attach Replicator Annotators to the Camera's Render Product ---
        render_product_path = self.camera.get_render_product_path()

        self.sem_seg = rep.AnnotatorRegistry.get_annotator("semantic_segmentation")
        self.sem_seg.attach([render_product_path])

        self.normals_annotator = rep.AnnotatorRegistry.get_annotator("normals")
        self.normals_annotator.attach([render_product_path])

        self.rgb_annotator = rep.AnnotatorRegistry.get_annotator("rgb")
        self.rgb_annotator.attach([render_product_path])

        self.depth_annotator = rep.AnnotatorRegistry.get_annotator("distance_to_image_plane")
        self.depth_annotator.attach([render_product_path])

        # --- 3. UDP Setup & State ---
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.is_processing = False
        self._shadow_diag_logged = False

        physical_properties = {
            "ground": [295, 0.93],    # Matches soil.
            # "road": [303, 0.95],      # Asphalt gets significantly hotter than soil.
            "road": [299, 0.95],      # Asphalt gets significantly hotter than soil.

            # --- Vegetation ---
            "bush": [293, 0.96],
            "leaves": [293, 0.97],
            "tree": [294, 0.94],
            "grass": [293, 0.958],
            "vegetation": [293, 0.958],
            "soil": [288, 0.914],

            # --- Hazards & Man-made ---
            "mine": [302, 0.93],      # Surface-laid TM-62M (painted steel).
            "car": [305, 0.90],
            "building": [296, 0.92],

            # --- People & Props ---
            "human": [301, 0.98],
            "cloth": [296, 0.95],

            # --- Debug ---
            "tag": [300, 0.934],
            "UNLABELED": [273, 0.0]   # Renders as a pitch-black "void".
        }

        # Radiance kept in float (no quantization) so per-frame AGC can stretch the range.
        self.radiance_map = {}
        for class_name, props in physical_properties.items():
            temp_k, emissivity = props
            self.radiance_map[class_name] = calculate_radiance(temp_k, emissivity)
            carb.log_info(f"Thermal Map - {class_name}: {self.radiance_map[class_name]:.3f}")

        # Air background radiance for distance haze mixing.
        air_temp = THERMAL_PARAMS["air_temp_K"]
        self.rad_air = calculate_radiance(air_temp, THERMAL_PARAMS["air_emissivity"])

        # Sun direction (unit vector pointing toward the sun, Z-up).
        el = np.deg2rad(THERMAL_PARAMS["sun_elevation_deg"])
        az = np.deg2rad(THERMAL_PARAMS["sun_azimuth_deg"])
        self.sun_dir = np.array(
            [np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)], dtype=np.float32
        )
        self.sun_dir = self.sun_dir / np.linalg.norm(self.sun_dir)

        # Pre-computed sensor artifacts (resolution-dependent).
        rng = np.random.default_rng(0)
        self.fixed_pattern = rng.normal(
            0, THERMAL_PARAMS["fixed_pattern_sigma"], (self.height, self.width)
        ).astype(np.float32)
        self.dead_pixels = rng.random((self.height, self.width)) < THERMAL_PARAMS["dead_pixel_fraction"]
        self.texture_base = rng.standard_normal(
            (THERMAL_PARAMS["texture_res"], THERMAL_PARAMS["texture_res"])
        ).astype(np.float32)

        # AGC temporal-smoothing state (prevents frame-to-frame brightness flicker).
        self._agc_lo = None
        self._agc_hi = None

    @staticmethod
    def _unwrap(annotator_data):
        """Replicator sometimes wraps annotator output in a dict."""
        if isinstance(annotator_data, dict):
            return annotator_data.get("data", annotator_data)
        return annotator_data

    def step_and_send(self):
        """Passively reads data without fighting the Isaac Sim timeline."""
        if self.is_processing:
            return

        data = self.sem_seg.get_data()

        if not data or "data" not in data or data["data"].size == 0:
            return

        self.is_processing = True
        threading.Thread(target=self._process_and_send, args=(data,), daemon=True).start()

    def _process_and_send(self, data):
        """Background thread for image processing and network transmission."""
        try:
            semantic_image = data["data"]
            id_to_labels = data["info"]["idToLabels"]

            H, W = self.height, self.width
            ids = np.squeeze(semantic_image).astype(np.int32).ravel()
            if ids.size == 0:
                return

            # 1) Base radiance from semantics (vectorized lookup).
            max_id = int(ids.max())
            if max_id > 100000:
                return
            lookup = np.zeros(max_id + 1, dtype=np.float32)
            for obj_id, labels in id_to_labels.items():
                class_name = labels.get("class", "UNLABELED")
                rad = self.radiance_map.get(class_name)
                if rad is None and class_name in MINE_LABEL_NAMES:
                    rad = self.radiance_map["mine"]
                lookup[int(obj_id)] = rad if rad is not None else self.radiance_map["UNLABELED"]
            rad = lookup[ids].reshape(H, W)

            # 2) Sun/orientation shading from surface normals (Lambertian).
            shade_norm = np.full((H, W), 1.0, dtype=np.float32)
            normals = self._normals_field()
            if normals is not None:
                lambert = np.maximum(0.0, np.sum(normals * self.sun_dir, axis=-1))
                shade_norm = THERMAL_PARAMS["shading_floor"] + (1 - THERMAL_PARAMS["shading_floor"]) * lambert

            # 3) Cast shadows (low-pass luminance from RGB), applied to all pixels.
            shadow = self._shadow_multiplier()

            rad *= np.clip(shade_norm * shadow, 0.0, 1.0)

            # 4) Distance haze toward air radiance.
            depth = self._depth_field()
            if depth is not None:
                atten = np.exp(-np.clip(depth, 0.0, None) * THERMAL_PARAMS["atmo_attenuation"])
                rad = rad * atten + self.rad_air * (1 - atten)

            # 5) Correlated thermal texture (breaks up flat regions).
            if THERMAL_PARAMS["texture_amp"] > 0:
                noise = cv2.resize(self.texture_base, (W, H), interpolation=cv2.INTER_CUBIC)
                rad *= (1.0 + THERMAL_PARAMS["texture_amp"] * noise)

            # Hotspot speckle.
            if THERMAL_PARAMS["speckle_fraction"] > 0:
                speckle = np.random.random((H, W)) < THERMAL_PARAMS["speckle_fraction"]
                dynamic = float(rad.max() - rad.min() + 1e-6)
                rad += speckle * (THERMAL_PARAMS["speckle_gain"] * dynamic)

            # 6) Per-frame AGC -> uint8 grayscale (full dynamic range each frame).
            grayscale_thermal = self._agc(rad)

            # 7) Sensor effects (readout noise, fixed pattern, dead pixels, blur).
            grayscale_thermal = self._apply_sensor_effects(grayscale_thermal)

            # INFO: NOISE (legacy)
            # noise = np.random.normal(0, 5, grayscale_thermal.shape).astype(np.uint8)
            # grayscale_thermal = cv2.add(grayscale_thermal, noise)
            # grayscale_thermal = cv2.GaussianBlur(grayscale_thermal, (5, 5), 0)

            # thermal_bgr = cv2.applyColorMap(grayscale_thermal, cv2.COLORMAP_INFERNO)

            _, buffer = cv2.imencode('.jpg', grayscale_thermal, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            jpg_bytes = buffer.tobytes()

            total_len = len(jpg_bytes)
            num_chunks = (total_len + MAX_DGRAM - 1) // MAX_DGRAM

            for i in range(num_chunks):
                start = i * MAX_DGRAM
                end = min(start + MAX_DGRAM, total_len)
                chunk = jpg_bytes[start:end]

                header = i.to_bytes(2, 'big') + num_chunks.to_bytes(2, 'big')
                self.sock.sendto(header + chunk, (self.ip, self.port))

        except Exception as e:
            carb.log_error(f"Thermal Processing Error: {e}")
        finally:
            self.is_processing = False

    def _normals_field(self):
        """World-space normals (H, W, 3) normalized to unit length, or None."""
        try:
            normals = self._unwrap(self.normals_annotator.get_data())
            if normals is None:
                return None
            normals = np.squeeze(normals)
            if normals.ndim != 3 or normals.shape[-1] != 3:
                return None
            if np.issubdtype(normals.dtype, np.integer):
                normals = (normals.astype(np.float32) / 255.0) * 2.0 - 1.0
            else:
                normals = normals.astype(np.float32)
            norm = np.linalg.norm(normals, axis=-1, keepdims=True)
            norm[norm < 1e-6] = 1.0
            return normals / norm
        except Exception as e:
            carb.log_warn(f"Thermal: normals unavailable ({e})")
            return None

    def _shadow_multiplier(self):
        """Cast-shadow multiplier in [0, 1] (1 = fully lit) for terrain classes.

        Isaac Sim 5.1's `occlusion` annotator only reports per-pixel object
        occlusion (instanceId/semanticId/occlusionRatio), not sun shadows, so we
        use a low-pass filtered luminance from the RGB annotator as the shadow
        proxy. The large blur suppresses albedo/color detail and keeps only the
        large-scale lighting variations (sun vs. cast shadow).
        """
        try:
            rgb = self._unwrap(self.rgb_annotator.get_data())
            if rgb is None:
                return np.ones((self.height, self.width), dtype=np.float32)
            rgb = np.squeeze(rgb).astype(np.float32)
            if rgb.ndim != 3 or rgb.shape[-1] < 3:
                return np.ones((self.height, self.width), dtype=np.float32)

            lum = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
            sigma = THERMAL_PARAMS["shadow_lowpass_sigma"]
            if sigma > 0:
                lum = cv2.GaussianBlur(lum, (0, 0), sigma)

            norm_ref = np.percentile(lum, THERMAL_PARAMS["shadow_norm_percentile"])
            lum_norm = lum / max(float(norm_ref), 1.0)
            lum_norm = np.clip(lum_norm, 0.0, 1.0)

            if not self._shadow_diag_logged:
                carb.log_info("Thermal: using RGB luminance low-pass for cast shadows.")
                self._shadow_diag_logged = True

            return THERMAL_PARAMS["shadow_floor"] + (1.0 - THERMAL_PARAMS["shadow_floor"]) * lum_norm
        except Exception as e:
            carb.log_warn(f"Thermal: shadow unavailable ({e})")
            return np.ones((self.height, self.width), dtype=np.float32)

    def _depth_field(self):
        """Depth to image plane in meters, or None."""
        try:
            depth = self._unwrap(self.depth_annotator.get_data())
            if depth is None:
                return None
            depth = np.squeeze(depth).astype(np.float32)
            if depth.ndim != 2:
                return None
            return depth
        except Exception as e:
            carb.log_warn(f"Thermal: depth unavailable ({e})")
            return None

    def _agc(self, rad):
        """Per-frame percentile stretch to uint8, with temporal smoothing.

        The percentile bounds are smoothed with an exponential moving average so
        the overall brightness adapts to the scene but doesn't flicker frame to
        frame. Set `agc_alpha` to 0.0 for a completely fixed mapping.
        """
        lo, hi = THERMAL_PARAMS["agc_percentiles"]
        p_lo, p_hi = np.percentile(rad, [lo, hi])
        if p_hi - p_lo < 1e-6:
            p_lo, p_hi = float(rad.min()), float(rad.max())

        alpha = THERMAL_PARAMS["agc_alpha"]
        if self._agc_lo is None:
            self._agc_lo, self._agc_hi = float(p_lo), float(p_hi)
        else:
            self._agc_lo = alpha * p_lo + (1.0 - alpha) * self._agc_lo
            self._agc_hi = alpha * p_hi + (1.0 - alpha) * self._agc_hi

        img = (rad - self._agc_lo) / (self._agc_hi - self._agc_lo + 1e-9)
        img = np.clip(img, 0.0, 1.0)
        return (img * 255.0).astype(np.uint8)

    def _apply_sensor_effects(self, img):
        """Readout noise, fixed-pattern bias, dead pixels, and low-MTF blur."""
        img = img.astype(np.float32)
        if THERMAL_PARAMS["fixed_pattern_sigma"] > 0:
            img += self.fixed_pattern
        if THERMAL_PARAMS["readout_noise_sigma"] > 0:
            img += np.random.normal(0, THERMAL_PARAMS["readout_noise_sigma"], img.shape).astype(np.float32)
        if THERMAL_PARAMS["dead_pixel_fraction"] > 0:
            img[self.dead_pixels] = 0.0
        out = np.clip(img, 0, 255).astype(np.uint8)
        sigma = THERMAL_PARAMS["sensor_blur_sigma"]
        if sigma > 0:
            out = cv2.GaussianBlur(out, (0, 0), sigma)
        return out
