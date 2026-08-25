"""
| File: tuning.py
| Description: Gradient-descent parameter tuner for multirotor physics.
|              Compares replay state logs against an original state log and
|              adjusts tunable parameters (drag coefficients, rolling
|              moment coefficient) to minimize trajectory error.
|
|              Driven by mission_helper._on_app_update via tick(frame_dt).
|              Maintains an internal state machine so that each tuning
|              iteration automatically: starts a replay, waits for the
|              motor CSV to finish + drone to disarm, computes the
|              error vs. the original state log, updates best, then
|              computes the next gradient step and starts the next
|              replay after a configurable cooldown.
| License: BSD-3-Clause.
"""
import os
import json
import csv
import glob
import math
import time
import numpy as np
import carb
from datetime import datetime

from pegasus.simulator.logic.dynamics.linear_drag import LinearDrag


class ParameterTuner:
    """Drives a gradient-descent loop that tunes multirotor physics parameters
    by comparing a replayed state log against an original state log.

    Tunable parameters (extensible):
        - drag: 3-element list [dx, dy, dz] (linear drag coefficients)
        - rolling_moment_coefficient: float (single scalar applied to all rotors)

    The tuner is driven frame-by-frame via tick(frame_dt) from the simulation's
    on_app_update loop. It does not block the main thread; each iteration
    completes asynchronously across many frames.
    """

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    FIXED_CONFIG_DIR = "/home/geekscode/ros2_ws/src/my_drone_pkg/my_drone_pkg"
    FIXED_CONFIG_PATH = os.path.join(FIXED_CONFIG_DIR, f"tuning/tuning_history_{timestamp}.json")
    STATE_LOGS_DIR = os.path.join(FIXED_CONFIG_DIR, "state_logs")

    # PARAMETER_KEYS = ["drag", "rolling_moment_coefficient"]
    PARAMETER_KEYS = ["drag"]

    # Internal states
    S_IDLE = "idle"
    S_EPISODE_RUNNING = "episode_running"
    S_EPISODE_FINISHED = "episode_finished"
    S_COMPUTING = "computing"
    S_COOLDOWN = "cooldown"
    S_PERTURBING = "perturbing"  # sub-state for gradient replays

    def __init__(self, mission_helper):
        self._mission_helper = mission_helper
        self._running = False

        # Configuration (set on start)
        self._original_log_path = ""
        self._motor_log_path = ""
        self._max_iterations = 100
        self._threshold = 0.00001
        # Adam step size (note: Adam step is normalized per-parameter, so the
        # meaningful range is roughly 1e-4 .. 1e-1, default tuned for drag values
        # in the 0.01 .. 0.5 range).
        self._learning_rate = 0.01
        self._adam_beta1 = 0.9
        self._adam_beta2 = 0.999
        self._adam_eps = 1e-8
        self._error_method = "euclidean"
        # self._perturbation_eps = 0.01
        self._perturbation_eps = 0.01
        self._cooldown_seconds = 2.0
        self._convergence_patience = 3
        self._allow_negative = False

        # State machine
        self._state = self.S_IDLE
        self._cooldown_remaining = 0.0
        self._frames_since_disarm = 0
        self._consecutive_below_threshold = 0
        self._last_iteration_error = None

        # Gradient sub-replay bookkeeping
        self._gradient_pending = None  # list of (flat_key, perturbed_params, error_to_record)
        self._gradient_index = 0
        self._base_params_for_gradient = None
        self._base_error_for_gradient = None
        self._gradient_flat_keys = []  # list of (key, idx_or_None)

        # Iteration progress
        self._current_iteration = 0
        # self._current_parameters = {
        #     "drag": [0.1, 0.1, 0.0],
        #     "rolling_moment_coefficient": 1e-6,
        # }

        self._current_parameters = {
            # "drag": [0.05, 0.05, 0.0],
            "drag": [0.15, 0.15, 0.1],
            # "rolling_moment_coefficient": 1e-6,
        }

        self._apply_parameters_to_physics(self._current_parameters)

        self._best_error = float("inf")
        self._best_parameters = dict(self._current_parameters)
        self._replays_paths = []
        self._history = []
        self._gradient_samples = []  # for forward-difference samples (separate from main history)

        # Adam optimizer state (per-parameter running moments + timestep).
        # Persisted to JSON so a paused/resumed run continues smoothly.
        self._adam_m = {}  # first moment, e.g. {"drag": [0.0, 0.0, 0.0]}
        self._adam_v = {}  # second moment
        self._adam_t = 0   # timestep counter

        # Live state for UI display
        self._live_parameters = None  # actual parameters currently in physics
        self._last_error = None  # most recently computed error (any source)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def is_running(self) -> bool:
        return self._running

    def start(
        self,
        original_log_path: str,
        motor_log_path: str,
        max_iterations: int,
        threshold: float,
        learning_rate: float,
        error_method: str,
        cooldown_seconds: float = 2.0,
        perturbation_eps: float = 0.01,
        allow_negative: bool = False,
    ):
        """Begin a fresh tuning session. Loads any existing history from disk
        for the same original log so previous progress is preserved.
        """
        if self._running:
            carb.log_warn("[tuning] Already running; ignoring start()")
            return

        if not os.path.exists(original_log_path):
            carb.log_error(f"[tuning] Original log not found: {original_log_path}")
            return
        if not os.path.exists(motor_log_path):
            carb.log_error(f"[tuning] Motor log not found: {motor_log_path}")
            return

        self._original_log_path = original_log_path
        self._motor_log_path = motor_log_path
        self._max_iterations = int(max_iterations)
        self._threshold = float(threshold)
        self._learning_rate = float(learning_rate)
        self._error_method = "dtw" if error_method == "dtw" else "euclidean"
        self._cooldown_seconds = max(0.0, float(cooldown_seconds))
        self._perturbation_eps = float(perturbation_eps)
        self._allow_negative = bool(allow_negative)
        self._consecutive_below_threshold = 0
        self._last_iteration_error = None
        self._last_error = None

        self.first_episode = True

        existing = self._load_history()
        if (
            existing
            and existing.get("original_log_path") == original_log_path
            and existing.get("error_method") == self._error_method
        ):
            self._current_iteration = int(existing.get("current_iteration", 0))
            self._current_parameters = existing.get(
                "current_parameters", self._read_current_physics_params()
            )
            self._best_error = float(existing.get("best_error", float("inf")))
            self._best_parameters = existing.get(
                "best_parameters", dict(self._current_parameters)
            )
            self._replays_paths = list(existing.get("replays_paths", []))
            self._history = list(existing.get("history", []))
            self._gradient_samples = list(existing.get("gradient_samples", []))
            # Restore Adam state if present (older JSONs may not have it)
            self._adam_t = int(existing.get("adam_t", 0))
            self._adam_m = existing.get("adam_m", {}) or {}
            self._adam_v = existing.get("adam_v", {}) or {}
        else:
            self._current_iteration = 0
            self._current_parameters = self._read_current_physics_params()
            self._best_error = float("inf")
            self._best_parameters = {
                k: list(v) if isinstance(v, list) else v
                for k, v in self._current_parameters.items()
            }
            self._replays_paths = []
            self._history = []
            self._gradient_samples = []

        self._running = True
        self._state = self.S_IDLE
        self._gradient_pending = None
        self._save_history()
        self._update_status_labels()

        carb.log_warn(
            f"[tuning] Started. orig={original_log_path} motor={motor_log_path} "
            f"max_iter={self._max_iterations} lr={self._learning_rate} "
            f"method={self._error_method} cooldown={self._cooldown_seconds}s"
        )

    def stop(self):
        """Halt the tuning loop. Leaves current parameters in place."""
        if not self._running:
            return
        self._running = False
        self._state = self.S_IDLE
        self._gradient_pending = None
        self._save_history()
        self._update_status_labels()
        carb.log_warn("[tuning] Stopped by user.")

    def apply_best_parameters(self):
        """Copy best_parameters into the live multirotor and helper state."""
        if not self._best_parameters:
            carb.log_warn("[tuning] No best parameters to apply.")
            return

        drag = self._best_parameters.get("drag", [0.1, 0.1, 0.0])
        rmc = self._best_parameters.get("rolling_moment_coefficient", 1e-6)

        try:
            multirotor = self._mission_helper.multirotors[0]
            multirotor._drag = LinearDrag(list(drag))
            # multirotor._thrusters._rolling_moment_coefficient = [
            #     float(rmc)
            # ] * multirotor._thrusters._num_rotors
        except Exception as e:
            carb.log_error(f"[tuning] Failed to apply best parameters: {e}")
            return

        try:
            models = self._mission_helper._window.drag_value_models
            if len(models) >= 3:
                models[0].set_value(float(drag[0]))
                models[1].set_value(float(drag[1]))
                models[2].set_value(float(drag[2]))
        except Exception:
            pass

        self._update_status_labels()
        carb.log_warn(f"[tuning] Applied best parameters: drag={drag} rmc={rmc}")

    # ------------------------------------------------------------------
    # Main driver: tick(frame_dt) called from mission_helper._on_app_update
    # ------------------------------------------------------------------
    def tick(self, frame_dt: float):
        if not self._running:
            return
        
        # carb.log_warn(f"current status: {self._state}")

        if self._state == self.S_IDLE:
            self._handle_idle()
        elif self._state == self.S_EPISODE_RUNNING:
            if self.first_episode:
                self._handle_idle()
                self.first_episode = False
            self._handle_episode_running()
        elif self._state == self.S_EPISODE_FINISHED:
            self._handle_episode_finished()
        elif self._state == self.S_COMPUTING:
            self._handle_computing()
        elif self._state == self.S_COOLDOWN:
            self._handle_cooldown(frame_dt)
        elif self._state == self.S_PERTURBING:
            self._handle_perturbing(frame_dt)

    # ------------------------------------------------------------------
    # State handlers
    # ------------------------------------------------------------------
    def _handle_idle(self):
        if self._gradient_pending is not None and self._gradient_index < len(self._gradient_pending):
            _, perturbed_params, _ = self._gradient_pending[self._gradient_index]
            self._apply_parameters_to_physics(perturbed_params)
        else:
            self._apply_parameters_to_physics(self._current_parameters)

    def _handle_episode_running(self):
        # Detect "replay finished": motor CSV exhausted + motors off
        helper = self._mission_helper
        replay_done = (
            not getattr(helper, "_motor_replay_mode", False)
            and not getattr(helper, "_motor_csv_loaded", False)
            and not getattr(helper, "motor_record_armed", False)
        )
        if replay_done:
            self._frames_since_disarm += 1
            # Wait a small number of frames to ensure state log is written
            if self._frames_since_disarm >= 5:
                self._state = self.S_EPISODE_FINISHED
                self._frames_since_disarm = 0
        else:
            self._frames_since_disarm = 0


    def _handle_episode_finished(self):
        self._state = self.S_COMPUTING


    def _handle_computing(self):
        replay_log = self._find_latest_state_log()

        if replay_log is None:
            carb.log_warn("[tuning] No state log found; skipping iteration.")
            self._schedule_next_iteration_or_stop()
            return

        try:
            error = self._compute_error(replay_log, self._original_log_path, self._error_method)
        except Exception as e:
            carb.log_error(f"[tuning] Error computation failed: {e}")
            self._running = False
            return

        # Track latest computed error for UI display (any source: base or perturbation)
        self._last_error = float(error)
        self._update_status_labels()

        # If we're in a gradient perturbation, store as a sample
        if self._gradient_pending is not None and self._gradient_index < len(self._gradient_pending):
            flat_key, perturbed_params, _ = self._gradient_pending[self._gradient_index]
            self._gradient_samples.append({
                "perturbation_key": self._flat_key_repr(flat_key),
                "parameters": {
                    k: list(v) if isinstance(v, list) else v
                    for k, v in perturbed_params.items()
                },
                "error": float(error),
                "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
            })
            self._gradient_index += 1
            # Move to cooldown then back to IDLE for the next perturbation
            self._cooldown_remaining = self._cooldown_seconds
            self._state = self.S_COOLDOWN
            self._save_history()
            self._update_status_labels()
            return

        # Otherwise: this is a real iteration
        if error < self._best_error:
            self._best_error = error
            self._best_parameters = {
                k: list(v) if isinstance(v, list) else v
                for k, v in self._current_parameters.items()
            }

        self._replays_paths.append(replay_log)
        self._history.append({
            "iteration": self._current_iteration,
            "parameters": {
                k: list(v) if isinstance(v, list) else v
                for k, v in self._current_parameters.items()
            },
            "replay_log_path": replay_log,
            "error": float(error),
            "error_method": self._error_method,
            "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        })

        carb.log_warn(
            f"[tuning] iter={self._current_iteration} error={error:.6f} "
            f"best_error={self._best_error:.6f} params={self._current_parameters}"
        )

        # Convergence check
        if self._last_iteration_error is not None:
            if abs(self._last_iteration_error - error) < self._threshold:
                self._consecutive_below_threshold += 1
            else:
                self._consecutive_below_threshold = 0
        self._last_iteration_error = error

        self._current_iteration += 1
        self._save_history()
        self._update_status_labels()

        if self._should_stop_loop():
            self._running = False
            carb.log_warn("[tuning] Stopping (max_iterations or convergence reached).")
            return

        # Schedule gradient: N+1 perturbations, each is its own replay
        self._start_gradient_estimation()
        self._cooldown_remaining = self._cooldown_seconds
        self._state = self.S_COOLDOWN

    def _handle_cooldown(self, frame_dt: float):
        # carb.log_warn("handle_cooldown")

        self._cooldown_remaining -= float(frame_dt)
        if self._cooldown_remaining <= 0:
            # If we just finished the last gradient perturbation, finalize
            if (
                self._gradient_pending is not None
                and self._gradient_index >= len(self._gradient_pending)
            ):
                self._finish_gradient_estimation()
            self._state = self.S_IDLE

    def _handle_perturbing(self, frame_dt: float):
        # Same as cooldown from a state machine perspective; we re-use the
        # gradient bookkeeping set in _start_gradient_estimation
        self._handle_cooldown(frame_dt)

    # ------------------------------------------------------------------
    # Gradient estimation orchestration
    # ------------------------------------------------------------------
    def _start_gradient_estimation(self):
        """After a main iteration, queue N+1 perturbation replays."""
        base_params = {
            k: (list(v) if isinstance(v, list) else v)
            for k, v in self._history[-1]["parameters"].items()
        }
        base_error = float(self._history[-1]["error"])

        flat_keys = []
        for key in self.PARAMETER_KEYS:
            val = base_params[key]
            if isinstance(val, list):
                for i in range(len(val)):
                    flat_keys.append((key, i))
            else:
                flat_keys.append((key, None))

        pending = []
        for flat_key in flat_keys:
            key, idx = flat_key
            perturbed = {
                k: (list(v) if isinstance(v, list) else v)
                for k, v in base_params.items()
            }
            if idx is None:
                perturbed[key] = float(base_params[key]) + self._perturbation_eps
            else:
                perturbed[key] = list(base_params[key])
                perturbed[key][idx] = float(base_params[key][idx]) + self._perturbation_eps
            pending.append((flat_key, perturbed, None))

        self._gradient_pending = pending
        self._gradient_index = 0
        self._base_params_for_gradient = base_params
        self._base_error_for_gradient = base_error
        self._gradient_flat_keys = flat_keys
        

    def _finish_gradient_estimation(self):
        """After all perturbation samples are collected, compute the gradient
        and update current_parameters.
        """
        if self._gradient_pending is None or self._gradient_index < len(self._gradient_pending):
            return

        eps = self._perturbation_eps
        base_error = self._base_error_for_gradient

        # Build a map from flat_key to perturbation error
        err_map = {}
        for sample in self._gradient_samples[-len(self._gradient_flat_keys):]:
            err_map[sample["perturbation_key"]] = sample["error"]

        # Compute gradient
        gradient = {}
        for flat_key in self._gradient_flat_keys:
            key, idx = flat_key
            flat_repr = self._flat_key_repr(flat_key)
            perturbed_error = err_map.get(flat_repr, base_error)
            grad_val = (perturbed_error - base_error) / eps
            if idx is None:
                gradient[key] = grad_val
            else:
                gradient.setdefault(key, [0.0] * len(self._base_params_for_gradient[key]))
                gradient[key][idx] = grad_val

        # Update current_parameters using Adam (per-parameter adaptive step)
        new_params = self._adam_update(self._base_params_for_gradient, gradient)
        self._current_parameters = new_params

        # Clamp to non-negative if allow_negative is False (allow zero)
        if not self._allow_negative:
            for key, val in self._current_parameters.items():
                if isinstance(val, list):
                    self._current_parameters[key] = [max(0.0, float(v)) for v in val]
                else:
                    self._current_parameters[key] = max(0.0, float(val))
            carb.log_warn(f"[tuning] Clamped non-negative: {self._current_parameters}")

        self._gradient_pending = None
        self._gradient_index = 0
        self._base_params_for_gradient = None
        self._base_error_for_gradient = None
        self._gradient_flat_keys = []

        self._save_history()
        self._update_status_labels()
        carb.log_warn(f"[tuning] Gradient step applied: new params={self._current_parameters}")

    @staticmethod
    def _flat_key_repr(flat_key) -> str:
        key, idx = flat_key
        return f"{key}[{idx}]" if idx is not None else f"{key}"

    def _ensure_adam_state_shape(self, params: dict):
        """Make sure the per-parameter moment buffers match the shape of
        `params`. Re-initializes a buffer (to zeros) if its length no longer
        matches (e.g. drag was extended from 3 to 4 axes).
        """
        for key, val in params.items():
            if isinstance(val, list):
                expected_len = len(val)
                if key not in self._adam_m or len(self._adam_m[key]) != expected_len:
                    self._adam_m[key] = [0.0] * expected_len
                if key not in self._adam_v or len(self._adam_v[key]) != expected_len:
                    self._adam_v[key] = [0.0] * expected_len
            else:
                if key not in self._adam_m or not isinstance(self._adam_m[key], float):
                    self._adam_m[key] = 0.0
                if key not in self._adam_v or not isinstance(self._adam_v[key], float):
                    self._adam_v[key] = 0.0

    def _adam_update(self, base_params: dict, gradient: dict) -> dict:
        """Apply one Adam optimizer step to `base_params` using `gradient`.

        Returns a new dict with the updated parameters. Side effects: mutates
        `self._adam_m`, `self._adam_v`, `self._adam_t` in-place. Lists in
        `base_params` are deep-copied; the caller's dict is not mutated.
        """
        # Defensive shape sync (in case PARAMETER_KEYS changed since last run)
        self._ensure_adam_state_shape(base_params)

        self._adam_t += 1
        t = self._adam_t
        beta1 = self._adam_beta1
        beta2 = self._adam_beta2
        eps_a = self._adam_eps
        lr = float(self._learning_rate)

        # Bias-correction denominators, computed once
        bc1 = 1.0 - (beta1 ** t)
        bc2 = 1.0 - (beta2 ** t)

        new_params = {}
        for key, val in base_params.items():
            g = gradient.get(key)
            if g is None:
                new_params[key] = list(val) if isinstance(val, list) else float(val)
                continue

            if isinstance(val, list):
                m_buf = self._adam_m[key]
                v_buf = self._adam_v[key]
                updated = []
                for i in range(len(val)):
                    gi = float(g[i])
                    m_buf[i] = beta1 * m_buf[i] + (1.0 - beta1) * gi
                    v_buf[i] = beta2 * v_buf[i] + (1.0 - beta2) * gi * gi
                    m_hat = m_buf[i] / bc1
                    v_hat = v_buf[i] / bc2
                    step = lr * m_hat / (math.sqrt(v_hat) + eps_a)
                    updated.append(float(val[i]) - step)
                new_params[key] = updated
            else:
                gi = float(g)
                self._adam_m[key] = beta1 * self._adam_m[key] + (1.0 - beta1) * gi
                self._adam_v[key] = beta2 * self._adam_v[key] + (1.0 - beta2) * gi * gi
                m_hat = self._adam_m[key] / bc1
                v_hat = self._adam_v[key] / bc2
                step = lr * m_hat / (math.sqrt(v_hat) + eps_a)
                new_params[key] = float(val) - step

        return new_params

    def _schedule_next_iteration_or_stop(self):
        """Used when an iteration is skipped (no state log found)."""
        if self._gradient_pending is not None and self._gradient_index < len(self._gradient_pending):
            # Skip this perturbation, count as base_error
            flat_key, perturbed_params, _ = self._gradient_pending[self._gradient_index]
            self._gradient_samples.append({
                "perturbation_key": self._flat_key_repr(flat_key),
                "parameters": {
                    k: list(v) if isinstance(v, list) else v
                    for k, v in perturbed_params.items()
                },
                "error": float("inf"),
                "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
            })
            self._gradient_index += 1
            self._cooldown_remaining = self._cooldown_seconds
            self._state = self.S_COOLDOWN
            return

        self._cooldown_remaining = self._cooldown_seconds
        self._state = self.S_COOLDOWN

    def _should_stop_loop(self) -> bool:
        if self._current_iteration >= self._max_iterations:
            return True
        # TEST: WITHOUT CONVERGENCE
        # if self._consecutive_below_threshold >= self._convergence_patience:
        #     return True
        return False

    # ------------------------------------------------------------------
    # Error computation
    # ------------------------------------------------------------------
    def _load_state_positions(self, path: str) -> np.ndarray:
        with open(path, "r") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        if not rows:
            raise ValueError(f"Empty state log: {path}")
        return np.array(
            [[float(r["pos_x"]), float(r["pos_y"]), float(r["pos_z"])] for r in rows],
            dtype=float,
        )

    def _euclidean_error(self, replay_pos: np.ndarray, original_pos: np.ndarray) -> float:
        n = min(len(replay_pos), len(original_pos))
        if n == 0:
            return float("inf")
        diff = replay_pos[:n] - original_pos[:n]
        return float(np.mean(np.linalg.norm(diff, axis=1)))

    def _dtw_error(self, replay_pos: np.ndarray, original_pos: np.ndarray) -> float:
        n = min(len(replay_pos), len(original_pos))
        if n == 0:
            return float("inf")
        r = replay_pos[:n]
        o = original_pos[:n]
        total = 0.0
        for axis in range(3):
            total += self._dtw_1d(r[:, axis], o[:, axis])
        return float(total)

    @staticmethod
    def _dtw_1d(x: np.ndarray, y: np.ndarray) -> float:
        n = len(x)
        m = len(y)
        dtw = np.full((n + 1, m + 1), np.inf, dtype=float)
        dtw[0, 0] = 0.0
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                cost = abs(x[i - 1] - y[j - 1])
                dtw[i, j] = cost + min(dtw[i - 1, j], dtw[i, j - 1], dtw[i - 1, j - 1])
        return float(dtw[n, m])

    def _compute_error(self, replay_log_path: str, original_log_path: str, method: str) -> float:
        replay_pos = self._load_state_positions(replay_log_path)
        original_pos = self._load_state_positions(original_log_path)
        if method == "dtw":
            return self._dtw_error(replay_pos, original_pos)
        return self._euclidean_error(replay_pos, original_pos)

    # ------------------------------------------------------------------
    # Physics access
    # ------------------------------------------------------------------
    def _read_current_physics_params(self) -> dict:
        drag = [0.1, 0.1, 0.0]
        rmc = 1e-6
        try:
            multirotor = self._mission_helper.multirotors[0]
            drag = list(np.diag(multirotor._drag._drag_coefficients))
            # rmc_list = list(multirotor._thrusters._rolling_moment_coefficient)
            # rmc = float(rmc_list[0]) if rmc_list else 1e-6
        except Exception as e:
            carb.log_warn(f"[tuning] Could not read current physics params: {e}")
        # return {"drag": drag, "rolling_moment_coefficient": rmc}
        return {"drag": drag}

    def _apply_parameters_to_physics(self, params: dict):
        try:
            multirotor = self._mission_helper.multirotors[0]
            drag = list(params.get("drag", [0.1, 0.1, 0.0]))
            # rmc = float(params.get("rolling_moment_coefficient", 1e-6))

            multirotor._drag = LinearDrag(drag)
            # multirotor._thrusters._rolling_moment_coefficient = [
            #     rmc
            # ] * multirotor._thrusters._num_rotors

            # carb.log_error(f"[tuning] _apply_parameters_to_physics {params}")
            try:
                models = self._mission_helper._window.drag_value_models
                if len(models) >= 3:
                    models[0].set_value(float(drag[0]))
                    models[1].set_value(float(drag[1]))
                    models[2].set_value(float(drag[2]))
            except Exception:
                pass

            # Track live parameters (deep copy) for UI display
            self._live_parameters = {
                k: (list(v) if isinstance(v, list) else float(v))
                for k, v in params.items()
            }
            self._update_status_labels()
        except Exception as e:
            carb.log_error(f"[tuning] _apply_parameters_to_physics failed: {e}")

    # ------------------------------------------------------------------
    # State log discovery
    # ------------------------------------------------------------------
    def _find_latest_state_log(self) -> str | None:
        pattern = os.path.join(self.STATE_LOGS_DIR, "vehicle_state_*.csv")
        candidates = glob.glob(pattern)
        if not candidates:
            return None
        candidates.sort(key=os.path.getmtime, reverse=True)
        return candidates[0]

    # ------------------------------------------------------------------
    # JSON history I/O
    # ------------------------------------------------------------------
    def _load_history(self) -> dict | None:
        if not os.path.exists(self.FIXED_CONFIG_PATH):
            return None
        try:
            with open(self.FIXED_CONFIG_PATH, "r") as f:
                return json.load(f)
        except Exception as e:
            carb.log_warn(f"[tuning] Failed to load history: {e}")
            return None

    def _save_history(self):
        os.makedirs(os.path.dirname(self.FIXED_CONFIG_PATH), exist_ok=True)
        payload = {
            "original_log_path": self._original_log_path,
            "motor_log_path": self._motor_log_path,
            "max_iterations": self._max_iterations,
            "threshold": self._threshold,
            "learning_rate": self._learning_rate,
            "error_method": self._error_method,
            "cooldown_seconds": self._cooldown_seconds,
            "allow_negative": self._allow_negative,
            "current_iteration": self._current_iteration,
            "current_parameters": {
                k: list(v) if isinstance(v, list) else v
                for k, v in self._current_parameters.items()
            },
            "best_error": (
                float(self._best_error)
                if self._best_error != float("inf")
                else None
            ),
            "best_parameters": {
                k: list(v) if isinstance(v, list) else v
                for k, v in self._best_parameters.items()
            },
            "replays_paths": list(self._replays_paths),
            "history": list(self._history),
            "gradient_samples": list(self._gradient_samples),
            # Adam optimizer state (so a paused/resumed run continues smoothly)
            "adam_t": int(self._adam_t),
            "adam_m": {
                k: (list(v) if isinstance(v, list) else float(v))
                for k, v in self._adam_m.items()
            },
            "adam_v": {
                k: (list(v) if isinstance(v, list) else float(v))
                for k, v in self._adam_v.items()
            },
        }
        try:
            with open(self.FIXED_CONFIG_PATH, "w") as f:
                json.dump(payload, f, indent=2)
        except Exception as e:
            carb.log_error(f"[tuning] Failed to save history: {e}")

    # ------------------------------------------------------------------
    # UI label updates
    # ------------------------------------------------------------------
    def _update_status_labels(self):
        helper = self._mission_helper
        current_text = self._format_params(self._current_parameters)
        best_text = self._format_params(self._best_parameters)
        live_text = self._format_params(self._live_parameters) if self._live_parameters else "N/A"
        best_error_text = (
            f"{self._best_error:.6f}" if self._best_error != float("inf") else "N/A"
        )
        current_error_text = (
            f"{self._last_error:.6f}" if self._last_error is not None else "N/A"
        )

        # Perturbation indicator: shown only when a gradient perturbation is currently active
        is_perturbing = (
            self._gradient_pending is not None
            and self._gradient_index < len(self._gradient_pending)
        )
        perturbation_text = "<perturbation>" if is_perturbing else "<default>"

        try:
            if helper._tuning_current_params_label is not None:
                helper._tuning_current_params_label.text = current_text
        except Exception:
            pass
        try:
            if helper._tuning_best_params_label is not None:
                helper._tuning_best_params_label.text = best_text
        except Exception:
            pass
        try:
            if helper._tuning_live_params_label is not None:
                helper._tuning_live_params_label.text = live_text
        except Exception:
            pass
        try:
            if helper._tuning_perturbation_indicator_label is not None:
                helper._tuning_perturbation_indicator_label.text = perturbation_text
        except Exception:
            pass
        try:
            if helper._tuning_best_error_label is not None:
                helper._tuning_best_error_label.text = best_error_text
        except Exception:
            pass
        try:
            if helper._tuning_current_error_label is not None:
                helper._tuning_current_error_label.text = current_error_text
        except Exception:
            pass
        try:
            if helper._tuning_best_params_label is not None:
                helper._tuning_best_params_label.text = best_text
        except Exception:
            pass
        try:
            if helper._tuning_best_error_label is not None:
                helper._tuning_best_error_label.text = best_error_text
        except Exception:
            pass

    @staticmethod
    def _format_params(params: dict) -> str:
        if not params:
            return "N/A"
        parts = []
        for k, v in params.items():
            if isinstance(v, list):
                parts.append(f"{k}=[" + ", ".join(f"{x:.4f}" for x in v) + "]")
            else:
                parts.append(f"{k}={v:.6f}")
        return ", ".join(parts)
