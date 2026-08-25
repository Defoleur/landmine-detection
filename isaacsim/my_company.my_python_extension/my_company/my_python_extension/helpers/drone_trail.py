import numpy as np
import carb


class DroneTrail:
    def __init__(self, debug_draw_interface, max_points=5000, min_step=0.1,
                 color=carb.ColorRgba(0, 1, 1, 1), thickness=2.0):
        self.debug_draw = debug_draw_interface
        self.max_points = max_points
        self.min_step = min_step
        self.color = color
        self.thickness = thickness
        self._positions = []

    def add_position(self, position):
        pos = np.array(position, dtype=float)

        if self._positions:
            last = self._positions[-1]
            dist = np.linalg.norm(pos - last)
            if dist > 10.0:
                self.clear()
            elif dist < self.min_step:
                return

        self._positions.append(pos)
        if len(self._positions) > self.max_points:
            self._positions.pop(0)

    def draw(self):
        # self.debug_draw.clear_lines()
        if len(self._positions) < 2:
            return

        pts = [carb.Float3(*p) for p in self._positions]
        n = len(pts) - 1
        start_points = pts[:-1]
        end_points = pts[1:]
        colors = [self.color] * n
        thicknesses = [self.thickness] * n
        self.debug_draw.draw_lines(start_points, end_points, colors, thicknesses)

    def clear(self):
        self._positions.clear()
