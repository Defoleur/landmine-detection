import carb
import math

class TrajectoryDrawer:
    """
    Handles all trajectory debug drawing logic.
    It takes the debug_draw interface on initialization.
    """
    
    def __init__(self, debug_draw_interface):
        self.debug_draw = debug_draw_interface
        
    def clear(self):
        """Clears all lines from the debug drawer."""
        self.debug_draw.clear_lines()

    def _get_path_data(self, received_positions):
        """
        Internal helper to parse the received positions list.
        Returns the start index and the list of polyline points.
        """
        if not received_positions:
            return None, None
            
        # Your logic to get the starting index
        index = int(received_positions[0][0] * 100) + 1
        
        # Get the polyline and convert to carb.Float3
        polyline = received_positions[index:]
        points = [carb.Float3(*p) for p in polyline]
        
        return index, points

    def _draw_path_segments(self, points, color=carb.ColorRgba(0, 1, 0, 1), thickness=2.0):
        """Internal helper to draw the main green trajectory path."""
        if len(points) < 2:
            return
            
        start_points = points[:-1]
        end_points = points[1:]
        colors = [color] * len(start_points)
        thicknesses = [thickness] * len(start_points)
        self.debug_draw.draw_lines(start_points, end_points, colors, thicknesses)

    def _draw_drone_to_path(self, drone_pos, path_points, color=carb.ColorRgba(1, 0, 0, 1), thickness=2.0):
        """Internal helper to draw the red line from drone to path start."""
        if drone_pos is None or not path_points:
            return
            
        start = carb.Float3(*drone_pos)
        end = path_points[0]
        self.debug_draw.draw_lines([start], [end], [color], [thickness])

    # --- Method 1: DRAW WITHOUT YAW ---
    
    def draw_simple_path(self, received_positions, drone_pos):
        """
        Draws just the trajectory and a line from the drone to its start.
        (Implementation of "DRAW WITHOUT YAW")
        """
        # self._clear()
        index, points = self._get_path_data(received_positions)
        
        if points is None:
            return
            
        self._draw_path_segments(points)
        self._draw_drone_to_path(drone_pos, points)

    # --- Method 2: DRAW TARGET YAW ---
    
    def draw_path_with_target_yaw(self, received_positions, yaws, drone_pos):
        """
        Draws the path, drone-to-path line, and a single yaw vector 
        at the drone's current position.
        (Implementation of "DRAW TARGET YAW")
        """
        # self._clear()
        index, points = self._get_path_data(received_positions)
        
        if points is None:
            return
            
        # Draw main path and drone line
        self._draw_path_segments(points)
        self._draw_drone_to_path(drone_pos, points)

        # Draw yaw direction line
        if drone_pos is not None and index < len(yaws):
            yaw_deg = yaws[index]  # take the first yaw as current drone heading
            yaw_rad = math.radians(yaw_deg - 270)
            yaw_length = 1.0  # meters

            dx = math.cos(yaw_rad)
            dy = math.sin(yaw_rad) * -1 # Your axis mapping

            start = carb.Float3(*drone_pos)
            end = carb.Float3(
                drone_pos[0] + dx * yaw_length,
                drone_pos[1] + dy * yaw_length,
                drone_pos[2]
            )
            self.debug_draw.draw_lines([start], [end], [carb.ColorRgba(0, 0, 1, 1)], [2.0])

    # --- Method 3: DRAW ALL YAWS ---

    def draw_path_with_all_yaws(self, received_positions, yaws, drone_pos):
        """
        Draws the path, drone-to-path line, a yaw vector for *every*
        point on the path, and a special yaw vector for the drone.
        (Implementation of "DRAW ALL YAWS")
        """
        # self._clear()
        index, points = self._get_path_data(received_positions)
        
        if points is None:
            return
        
        # Draw main path and drone line
        self._draw_path_segments(points)
        self._draw_drone_to_path(drone_pos, points)

        # --- Draw yaw direction lines for each point on the path ---
        if len(yaws) > index:
            yaw_length = 0.5  # shorter arrows for the path
            yaw_lines_start = []
            yaw_lines_end = []
            yaw_colors = []
            yaw_thicknesses = []

            for i, p in enumerate(points):
                if index + i >= len(yaws):
                    break
                yaw_deg = yaws[index + i]
                yaw_rad = math.radians(yaw_deg - 270)

                dx = math.cos(yaw_rad)
                dy = -math.sin(yaw_rad)  # inverted Y for coordinate system

                start = carb.Float3(p.x, p.y, p.z)
                end = carb.Float3(p.x + dx * yaw_length, p.y + dy * yaw_length, p.z)

                yaw_lines_start.append(start)
                yaw_lines_end.append(end)
                yaw_colors.append(carb.ColorRgba(0, 0, 1, 1))  # blue
                yaw_thicknesses.append(1.5)

            self.debug_draw.draw_lines(yaw_lines_start, yaw_lines_end, yaw_colors, yaw_thicknesses)

        # --- Draw current drone yaw direction (special color) ---
        if drone_pos is not None and index < len(yaws):
            yaw_deg = yaws[index]
            yaw_rad = math.radians(yaw_deg - 270)
            dx = math.cos(yaw_rad)
            dy = -math.sin(yaw_rad)
            yaw_length = 1.0 # longer arrow for the drone

            start = carb.Float3(*drone_pos)
            end = carb.Float3(
                drone_pos[0] + dx * yaw_length,
                drone_pos[1] + dy * yaw_length,
                drone_pos[2]
            )
            self.debug_draw.draw_lines([start], [end], [carb.ColorRgba(1, 0.6, 0, 1)], [2.5])