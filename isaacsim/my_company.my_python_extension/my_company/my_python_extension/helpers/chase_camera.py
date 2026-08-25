import numpy as np
import carb
import math
from pxr import Gf
from omni.isaac.sensor import Camera


class ChaseCamera:
    def __init__(self, prim_path="/World/camera_follow", resolution=(640, 480),
                 frequency=30, focal_length=3.0, clip_range=(0.01, 10000),
                 offset=(-6.0, 0.0, 3.0), smoothness=10.0, target_smoothness=2.0):
        self.prim_path = prim_path
        self.resolution = resolution
        self.frequency = frequency
        self.focal_length = focal_length
        self.clip_range = clip_range
        self.offset = np.array(offset, dtype=float)
        self.smoothness = smoothness
        self.target_smoothness = target_smoothness
        self._cam = None
        self._smoothed_target = None

    def initialize(self):
        if self.is_initialized():
            return
        try:
            self._cam = Camera(
                prim_path=self.prim_path,
                frequency=self.frequency,
                resolution=self.resolution
            )
            self._cam.set_focal_length(self.focal_length)
            self._cam.set_clipping_range(self.clip_range[0], self.clip_range[1])
            self._cam.initialize()
        except Exception as e:
            carb.log_error(f"ChaseCamera: Failed to initialize: {e}")
            self._cam = None

    def is_initialized(self):
        return self._cam is not None

    def update(self, target_position, dt=1.0/60.0):
        if not self._cam or target_position is None:
            return

        raw_pos = np.array(target_position)

        # Stage 1: heavy low-pass on raw target to kill physics jitter
        if self._smoothed_target is None:
            self._smoothed_target = raw_pos.copy()
        else:
            t_target = 1.0 - math.exp(-self.target_smoothness * dt)
            self._smoothed_target = self._smoothed_target * (1.0 - t_target) + raw_pos * t_target

        # Stage 2: camera position lerps toward smoothed target + offset
        curr_p, _ = self._cam.get_world_pose()
        des_p = self._smoothed_target + self.offset
        t_cam = 1.0 - math.exp(-self.smoothness * dt)
        new_p = curr_p * (1.0 - t_cam) + des_p * t_cam

        # Look at the smoothed target so orientation doesn't jitter
        self._cam.set_world_pose(new_p, self._look_at(new_p, self._smoothed_target))

    def reset(self):
        self._cam = None
        self._smoothed_target = None

    @staticmethod
    def _look_at(src, tgt):
        d = np.array(tgt) - np.array(src)
        n = np.linalg.norm(d)
        if n < 1e-6:
            return [1, 0, 0, 0]
        d /= n
        y = math.atan2(d[1], d[0])
        p = math.asin(np.clip(d[2], -1, 1))
        q = Gf.Quatf(math.cos(y/2)*math.cos(p/2), 0, math.sin(p/2), math.sin(y/2))
        return [q.GetReal(), q.GetImaginary()[0], -q.GetImaginary()[1], q.GetImaginary()[2]]
