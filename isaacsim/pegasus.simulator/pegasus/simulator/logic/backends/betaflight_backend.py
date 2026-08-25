"""
| File: betaflight_backend.py
| Author: Marcelo Jacinto (marcelo.jacinto@tecnico.ulisboa.pt)
| Description: File that implements the Mavlink Backend for communication/control with/of the vehicle simulation
| License: BSD-3-Clause. Copyright (c) 2023, Marcelo Jacinto. All rights reserved.
"""
__all__ = ["BetaflightBackend", "BetaflightBackendConfig"]

import carb
import time
import numpy as np
import csv
import math
from datetime import datetime
import omni.kit.app
from enum import Enum

from pegasus.simulator.logic.state import State
from pegasus.simulator.logic.backends.backend import Backend, BackendConfig
from pegasus.simulator.logic.backends.tools.betaflight_api import BetaflightApiPython  # adjust import path if needed

# REMOVE?
class PWMType(Enum):
    RAW = 1,
    PWM_OUTPUT = 2,
    DSHOT300 = 3


class SensorMsg:
    """
    An auxiliary data class where we write all the sensor data that is going to be sent through mavlink
    """

    def __init__(self):
        # IMU Data
        self.new_imu_data: bool = False
        self.received_first_imu: bool = False
        self.xacc: float = 0.0
        self.yacc: float = 0.0
        self.zacc: float = 0.0
        self.xgyro: float = 0.0
        self.ygyro: float = 0.0
        self.zgyro: float = 0.0

        # Baro Data
        self.new_bar_data: bool = False
        self.abs_pressure: float = 0.0
        self.pressure_alt: float = 0.0
        self.temperature: float = 0.0

        # Magnetometer Data
        self.new_mag_data: bool = False
        self.xmag: float = 0.0
        self.ymag: float = 0.0
        self.zmag: float = 0.0

        # Airspeed Data
        self.new_press_data: bool = False
        self.diff_pressure: float = 0.0

        # GPS Data
        self.new_gps_data: bool = False
        self.fix_type: int = 0
        self.latitude_deg: float = -999
        self.longitude_deg: float = -999
        self.altitude: float = -999
        self.eph: float = 1.0
        self.epv: float = 1.0
        self.velocity: float = 0.0
        self.velocity_north: float = 0.0
        self.velocity_east: float = 0.0
        self.velocity_down: float = 0.0
        self.cog: float = 0.0
        self.satellites_visible: int = 0

        # Vision Pose
        self.new_vision_data: bool = False
        self.vision_x: float = 0.0
        self.vision_y: float = 0.0
        self.vision_z: float = 0.0
        self.vision_roll: float = 0.0
        self.vision_pitch: float = 0.0
        self.vision_yaw: float = 0.0
        self.vision_covariance = (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)

        # Simulation State
        self.new_sim_state: bool = False
        self.sim_attitude = [1.0, 0.0, 0.0, 0.0]  # [w, x, y, z]
        self.sim_acceleration = [0.0, 0.0, 0.0]  # [x,y,z body acceleration]
        self.sim_angular_vel = [0.0, 0.0, 0.0]  # [roll-rate, pitch-rate, yaw-rate] rad/s
        self.sim_lat = 0.0  # [deg]
        self.sim_lon = 0.0  # [deg]
        self.sim_alt = 0.0  # [m]
        self.sim_ind_airspeed = 0.0  # Indicated air speed
        self.sim_true_airspeed = 0.0  # Indicated air speed
        self.sim_velocity_inertial = [0.0, 0.0, 0.0]  # North-east-down [m/s]

class ThrusterControl:
    """
    An auxiliary data class that saves the thrusters command data received via mavlink and
    scales them into individual angular velocities expressed in rad/s to apply to each rotor
    """

    def __init__(
        self,
        num_rotors: int = 4,
        input_offset=[0, 0, 0, 0],
        input_scaling=[0, 0, 0, 0],
        zero_position_armed=[100, 100, 100, 100],
    ):
        """Initialize the ThrusterControl object

        Args:
            num_rotors (int): The number of rotors that the actual system has 4.
            input_offset (list): A list with the offsets to apply to the rotor values received via mavlink. Defaults to [0, 0, 0, 0].
            input_scaling (list): A list with the scaling to apply to the rotor values received via mavlink. Defaults to [0, 0, 0, 0].
            zero_position_armed (list): Another list of offsets to apply to the rotor values received via mavlink. Defaults to [100, 100, 100, 100].
        """

        self.num_rotors: int = num_rotors

        self.armed = False

        # Values to scale and offset the rotor control inputs received from PX4
        assert len(input_offset) == self.num_rotors
        self.input_offset = input_offset

        assert len(input_scaling) == self.num_rotors
        self.input_scaling = input_scaling

        assert len(zero_position_armed) == self.num_rotors
        self.zero_position_armed = zero_position_armed

        # The actual speed references to apply to the vehicle rotor joints
        self._input_reference = [0.0 for i in range(self.num_rotors)]

    @property
    def input_reference(self):
        """A list of floats with the angular velocities in rad/s

        Returns:
            list: A list of floats with the angular velocities to apply to each rotor, expressed in rad/s
        """
        return self._input_reference

    def update_input_reference(self, controls):
        """Takes a list with the thrust controls received via mavlink and scales them in order to generated
        the equivalent angular velocities in rad/s

        Args:
            controls (list): A list of ints with thrust controls received via mavlink
        """

        # Check if the number of controls received is correct
        if len(controls) < self.num_rotors:
            carb.log_warn("Did not receive enough inputs for all the rotors")
            return

        if all(c == 0 for c in controls) and self.armed == True:
            self.armed = False

        if any(c > 0 for c in controls) and self.armed == False:
            self.armed = True

        # Update the desired reference for every rotor (and saturate according to the min and max values)
        for i in range(self.num_rotors):
            # Compute the actual velocity reference to apply to each rotor
            self._input_reference[i] = (controls[i] + self.input_offset[i]) * self.input_scaling[i] + \
                                       self.zero_position_armed[i]

    def zero_input_reference(self):
        """
        When this method is called, the input_reference is updated such that every rotor is stopped
        """
        self._input_reference = [0.0 for i in range(self.num_rotors)]


class BetaflightBackendConfig(BackendConfig):
    """
    An auxiliary data class used to store all the configurations for the mavlink communications.
    """

    def __init__(self, config: dict = {}):
        """
        Initialize the BetaflightBackendConfig class.

        Args:
            config (dict): Optional dictionary of parameters for MSP connection.
                           Missing keys will be filled with defaults.
        """
        self.config = config

        # Basic identity
        self.vehicle_id = self.config.get("vehicle_id", 0)

        # Connection setup
        self.connection_ip = self.config.get("connection_ip", "127.0.0.1")
        self.motor_port = self.config.get("motor_port", 9002)
        self.state_port = self.config.get("state_port", 9003)

        # Rotor control mapping
        self.num_rotors = self.config.get("num_rotors", 4)
        self.input_offset = self.config.get("input_offset", [0.0] * self.num_rotors)
        self.input_scaling = self.config.get("input_scaling", [1000.0] * self.num_rotors)
        self.zero_position_armed = self.config.get("zero_position_armed", [100.0] * self.num_rotors)

        # Update rate for control loop
        self.update_rate: float = self.config.get("update_rate", 250.0)  # Hz

class BetaflightBackend(Backend):
    """
    The MSP Backend used to receive the vehicle's state and sensor data from
    Betaflight via MSP, and to send control commands back to the flight controller.
    """

    def __init__(self, config: BetaflightBackendConfig = BetaflightBackendConfig()):
        """
        Initialize the BetaflightBackend.

        Args:
            config (BetaflightBackendConfig): The configuration for MSP communications.
        """
        super().__init__(config)

        self.config: BetaflightBackendConfig = config
        self._vehicle_id = self.config.vehicle_id

        # Sensor data received from Betaflight (e.g. IMU, GPS, barometer)
        self._sensor_data = SensorMsg()

        self._update_rate: float = self.config.update_rate
        self._time_step: float = 1.0 / self._update_rate

        # Rotor control commands to send to Betaflight
        self._rotor_data = ThrusterControl(
            self.config.num_rotors,
            self.config.input_offset,
            self.config.input_scaling,
            self.config.zero_position_armed
        )

        # Actuator state
        self._num_inputs = self.config.num_rotors
        self._input_reference = np.zeros((self._num_inputs,))

        self._input_offset = np.zeros((self._num_inputs,))
        self._input_scaling = np.zeros((self._num_inputs,))

        self._current_utime = 0

        self._api = BetaflightApiPython(motor_port=self.config.motor_port,
                                        state_port=self.config.state_port,
                                        udp_ip=self.config.connection_ip)

        self._log_time = time.time()

        self.custom_imu_enabled = False
        self.custom_imu_path = None

        self.custom_imu_data = []
        self.custom_imu_index = 0
        self._custom_imu_loaded = False
        self._reload_custom_imu_on_next_arm = True
        self._prev_armed = False

        self._arm_time_monotonic_ns = None
        self._custom_imu_base_timestamp_ns = 0

        # self.acc_scale = -9.80665 / 256.0
        # self.gyro_scale = -math.pi / 180.0
        # self.gyro_scale = -math.pi / 90.0

        # self.acc_scale = -0.038454
        # self.gyro_scale = -0.001063
        
        self.acc_scale = 1
        self.gyro_scale = 1

        self._was_armed = False
        self._live_imu_buffer = []

        self.motor_csv_data = []
        self.motor_csv_index = 0
        self.use_motor_csv = False
        self.motor_csv_loaded = False
        
        # Motor CSV replay timing tracking
        self._motor_replay_start_time_ns = 0
        self._motor_replay_index = 0
        self.motor_csv_timing_mode = False  # True = time-based, False = sequential
        self._motor_replay_wall_base = 0
        self._motor_csv_first_time = 0

        # --- Motor Recording Variables ---
        self._motor_was_armed = True
        self._live_motor_buffer = []

        self.update_disabled = False
        self.current_controls = [0, 0, 0, 0]

    def update_sensor(self, sensor_type: str, data):
        """
        Store the most recent sensor readings so they can be sent to Betaflight.
        Supported types: IMU, GPS, Barometer, Magnetometer.
        """
        if sensor_type == "IMU":
            self.update_imu_data(data)
        elif sensor_type == "GPS":
            self.update_gps_data(data)
        elif sensor_type == "Barometer":
            self.update_bar_data(data)
        # Not used by Betaflight
        elif sensor_type == "Magnetometer":
            self.update_mag_data(data)
        else:
            # Unsupported sensor type for MSP
            pass

    def _load_custom_imu_csv(self):
        """
        Loads IMU replay rows from self.custom_imu_path.
        Expected columns (preferred):
            timestamp,w_x,w_y,w_z,a_x,a_y,a_z
        Also supports aliases:
            xgyro,ygyro,zgyro,xacc,yacc,zacc
        """
        self.custom_imu_data = []
        self.custom_imu_index = 0
        self._custom_imu_loaded = False
        self._custom_imu_base_timestamp_ns = 0
        if not self.custom_imu_path:
            carb.log_warn("custom_imu_enabled=True but custom_imu_path is empty.")
            return False
        try:
            with open(self.custom_imu_path, "r", encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # timestamp (optional fallback to row index timing if absent)
                    ts_raw = row.get("timestamp", row.get("time_ns", None))
                    if ts_raw is None or str(ts_raw).strip() == "":
                        continue
                    # gyro aliases
                    xg = row.get("w_x", row.get("xgyro", None))
                    yg = row.get("w_y", row.get("ygyro", None))
                    zg = row.get("w_z", row.get("zgyro", None))
                    # accel aliases
                    xa = row.get("a_x", row.get("xacc", None))
                    ya = row.get("a_y", row.get("yacc", None))
                    za = row.get("a_z", row.get("zacc", None))
                    if xg is None or yg is None or zg is None or xa is None or ya is None or za is None:
                        continue
                    self.custom_imu_data.append({
                        "timestamp_ns": int(float(ts_raw)),
                        "xgyro": float(xg) * self.gyro_scale,
                        "ygyro": float(yg) * self.gyro_scale,
                        "zgyro": float(zg) * self.gyro_scale,
                        "xacc": float(xa) * self.acc_scale,
                        "yacc": float(ya) * self.acc_scale,
                        "zacc": float(za) * self.acc_scale,
                    })
            if len(self.custom_imu_data) == 0:
                carb.log_warn(f"No valid IMU rows parsed from {self.custom_imu_path}")
                return False
            # ensure monotonic by timestamp
            self.custom_imu_data.sort(key=lambda r: r["timestamp_ns"])
            self._custom_imu_base_timestamp_ns = self.custom_imu_data[0]["timestamp_ns"]
            self._custom_imu_loaded = True
            self._reload_custom_imu_on_next_arm = False
            carb.log_info(f"Loaded {len(self.custom_imu_data)} custom IMU rows from {self.custom_imu_path}")
            return True
        except Exception as e:
            carb.log_warn(f"Failed to load custom IMU CSV from {self.custom_imu_path}: {e}")
            return False

    def _get_custom_imu_row_for_elapsed_ns(self, elapsed_ns: int):
        """
        Returns the row that matches current arm-relative elapsed time.
        Holds last known row between timestamps and after end-of-file.
        """
        if not self.custom_imu_data:
            return None
        target_ts = self._custom_imu_base_timestamp_ns + max(0, int(elapsed_ns))
        # advance cursor while next row timestamp is <= target
        while self.custom_imu_index + 1 < len(self.custom_imu_data):
            next_ts = self.custom_imu_data[self.custom_imu_index + 1]["timestamp_ns"]
            if next_ts > target_ts:
                break
            self.custom_imu_index += 1
        return self.custom_imu_data[self.custom_imu_index]

    def _reset_custom_imu_for_next_mission(self):
        """
        Called on disarm to force reloading file on next arm.
        """
        self.custom_imu_data = []
        self.custom_imu_index = 0
        self._custom_imu_loaded = False
        self._reload_custom_imu_on_next_arm = True
        self._arm_time_monotonic_ns = None
        self._custom_imu_base_timestamp_ns = 0

    def _update_arm_state_and_reload_logic(self):
        """
        Tracks arm/disarm transitions and reloads custom IMU on arm if requested.
        """
        # Motor CSV replay start tracking
        if self.use_motor_csv:
            # Rising edge: disarmed -> armed
            if self._rotor_data.armed and not self._prev_armed:
                self._motor_replay_start_time_ns = time.time_ns()
                self._motor_replay_index = 0
                self._motor_replay_wall_base = 0  # Reset for time-based mode
                carb.log_warn(f"Motor CSV replay started. Timing mode: {self.motor_csv_timing_mode}")
            # Falling edge: armed -> disarmed
            if (not self._rotor_data.armed) and self._prev_armed:
                # Reset timing on disarm
                self._motor_replay_index = 0
                self._motor_replay_wall_base = 0
        
        if self.custom_imu_enabled:
            # Rising edge: disarmed -> armed
            if self._rotor_data.armed and not self._prev_armed:
                self._arm_time_monotonic_ns = time.monotonic_ns()
                if self._reload_custom_imu_on_next_arm:
                    ok = self._load_custom_imu_csv()
                    if not ok:
                        carb.log_warn("Custom IMU reload failed on arm; falling back to live IMU.")
            # Falling edge: armed -> disarmed
            if (not self._rotor_data.armed) and self._prev_armed:
                self._reset_custom_imu_for_next_mission()
        self._prev_armed = self._rotor_data.armed

    def update_imu_data(self, data):
        """
        Update IMU values.
        If custom IMU is enabled AND armed, override with CSV row selected by elapsed arm time.
        """

        use_custom = (
            self.custom_imu_enabled
            and self._rotor_data.armed
            and self._custom_imu_loaded
            and self._arm_time_monotonic_ns is not None
        )
        # carb.log_warn(f"Custom IMU Checks: armed - {self._rotor_data.armed}, custom_imu_enabled - {self.custom_imu_enabled}, custom_imu_loaded - {self._custom_imu_loaded}, arm_time_monotonic_ns - {self._arm_time_monotonic_ns}")
        if use_custom:
            elapsed_ns = time.monotonic_ns() - self._arm_time_monotonic_ns
            row = self._get_custom_imu_row_for_elapsed_ns(elapsed_ns)
            carb.log_warn(f"Using custom IMU! Data: {row}")
            if row is not None:
                accel = data["linear_acceleration"]
                self._sensor_data.xacc = row["xacc"]
                self._sensor_data.yacc = row["yacc"]
                self._sensor_data.zacc = row["zacc"]
                self._sensor_data.xgyro = row["xgyro"]
                self._sensor_data.ygyro = row["ygyro"]
                self._sensor_data.zgyro = row["zgyro"]
            else:
                # Fallback to live sensor if row lookup fails
                accel = data["linear_acceleration"]
                self._sensor_data.xacc = accel[0]
                self._sensor_data.yacc = accel[1]
                self._sensor_data.zacc = accel[2]
                self._sensor_data.xgyro = data["angular_velocity"][0]
                self._sensor_data.ygyro = data["angular_velocity"][1]
                self._sensor_data.zgyro = data["angular_velocity"][2]
        else:
            # Default: use live simulated IMU
            accel = data["linear_acceleration"]
            # carb.log_warn(f"Using live IMU! Data: {data['linear_acceleration']}, {data['angular_velocity']}")
            self._sensor_data.xacc = accel[0]
            self._sensor_data.yacc = accel[1]
            self._sensor_data.zacc = accel[2]
            self._sensor_data.xgyro = data["angular_velocity"][0]
            self._sensor_data.ygyro = data["angular_velocity"][1]
            self._sensor_data.zgyro = data["angular_velocity"][2]
        self._sensor_data.new_imu_data = True
        self._sensor_data.received_first_imu = True

        # # ==========================================
        # # --- LIVE IMU RECORDING LOGIC ---
        # # ==========================================
        currently_armed = self._rotor_data.armed

        # 1. Detect ARM transition (Disarmed -> Armed)
        if currently_armed and not self._was_armed:
            self._live_imu_buffer.clear()
            carb.log_warn("Drone ARMED. Started recording live IMU data with absolute timestamps.")

        # 2. Collect data while ARMED
        if currently_armed:
            # THE FIX: Grab the absolute system time instead of calculating elapsed time
            absolute_ns = time.time_ns()
            
            live_accel = data["linear_acceleration"]
            live_gyro = data["angular_velocity"]

            self._live_imu_buffer.append({
                "time_ns": absolute_ns,
                "xacc": live_accel[0],
                "yacc": live_accel[1],
                "zacc": live_accel[2],
                "xgyro": live_gyro[0],
                "ygyro": live_gyro[1],
                "zgyro": live_gyro[2]
            })

        # 3. Detect DISARM transition (Armed -> Disarmed)
        if not currently_armed and self._was_armed:
            self._save_imu_buffer_to_csv()

        # Save state for the next tick
        self._was_armed = currently_armed

    def _save_imu_buffer_to_csv(self):
        """Writes the recorded IMU buffer to a CSV file upon disarm."""
        if not self._live_imu_buffer:
            return

        # Generate a unique filename so you don't overwrite previous flights
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"/home/geekscode/ros2_ws/src/my_drone_pkg/my_drone_pkg/live_imu_logs/live_imu_record_{timestamp}.csv"

        try:
            with open(filename, mode='w', newline='') as csvfile:
                fieldnames = ["time_ns", "xacc", "yacc", "zacc", "xgyro", "ygyro", "zgyro"]
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

                writer.writeheader()
                writer.writerows(self._live_imu_buffer)

            carb.log_warn(f"Drone DISARMED. Saved {len(self._live_imu_buffer)} IMU records to {filename}")
        except Exception as e:
            carb.log_error(f"Failed to save IMU CSV: {e}")
        finally:
            # Free the memory for the next flight
            self._live_imu_buffer.clear()

    def load_motor_csv(self, filepath: str, step: int = 1):
        """Loads motor RPM/PWM data, normalizes it globally, and samples every Nth row."""
        self.motor_csv_data.clear()
        self.motor_csv_index = 0
        self._motor_replay_index = 0
        
        try:
            # Temporary buffer for raw data
            raw_data_buffer = []
            
            # Trackers for the absolute highest and lowest values across ALL motors
            global_min = float('inf')
            global_max = float('-inf')
            
            with open(filepath, 'r') as f:
                reader = csv.DictReader(f)
                
                for i, row in enumerate(reader):
                    # Skip the row if it's not a multiple of the step
                    if i % step != 0 and i != 0:
                        continue
                    
                    # 1. Read the timestamp if available
                    time_ns = int(row.get('time_ns', 0)) if 'time_ns' in row else 0
                    time_ns = int(float(row.get('timestamp', 0)) * 1e9) if 'timestamp' in row else time_ns
                    
                    # 2. Extract raw motor values
                    raw_m0 = float(row['motor0'])
                    raw_m1 = float(row['motor1'])
                    raw_m2 = float(row['motor2'])
                    raw_m3 = float(row['motor3'])
                    
                    # 3. Update the global minimum and maximum bounds
                    row_min = min(raw_m0, raw_m1, raw_m2, raw_m3)
                    row_max = max(raw_m0, raw_m1, raw_m2, raw_m3)
                    
                    if row_min < global_min: global_min = row_min
                    if row_max > global_max: global_max = row_max
                    
                    sim_time = float(row.get("sim_time", 0))
                    
                    # 4. Store raw data in the buffer
                    raw_data_buffer.append({
                        'time_ns': time_ns,
                        'motor0': raw_m0,
                        'motor1': raw_m1,
                        'motor2': raw_m2,
                        'motor3': raw_m3,
                        'sim_time': sim_time
                    })
            
            # 5. Prevent Division by Zero (if all motors in the CSV are stuck at 0)

            if global_min <= 1 and global_max <= 1:
                pwm_type = PWMType.RAW
                subtract_coef = 0
                norm_coef = 1
            elif global_min < 900:
                pwm_type = PWMType.DSHOT300
                subtract_coef = 0
                norm_coef = 2000.0
            else:
                pwm_type = PWMType.PWM_OUTPUT
                subtract_coef = 1000.0
                norm_coef = 1000.0
            
            # 6. Normalize the buffered data (0.0 to 1.0) and push to the final list
            for data in raw_data_buffer:
                self.motor_csv_data.append({
                    'time_ns': data['time_ns'],
                    'motor0': max(0.0, min(1.0, (data['motor0'] - subtract_coef) / norm_coef)),
                    'motor1': max(0.0, min(1.0, (data['motor1'] - subtract_coef) / norm_coef)),
                    'motor2': max(0.0, min(1.0, (data['motor2'] - subtract_coef) / norm_coef)),
                    'motor3': max(0.0, min(1.0, (data['motor3'] - subtract_coef) / norm_coef)),
                    'sim_time': data['sim_time']
                })
            
            if len(self.motor_csv_data) > 0:
                self.use_motor_csv = True
                carb.log_warn(f"Loaded {len(self.motor_csv_data)} commands. Scaled using Global Min: {global_min}, Global Max: {global_max}")
            else:
                carb.log_error("Motor CSV is empty.")
                
        except Exception as e:
            carb.log_error(f"Failed to load motor CSV: {e}")
            self.use_motor_csv = False

    def handle_control(self, controls):
        """
        Method that when received a control message, compute the simulated force that should be applied
        on each rotor of the vehicle

        Args:
            controls (list): A list of ints which contains the thrust_control received via mavlink
        """
        
        currently_armed = self._rotor_data.armed

        applied_controls = controls
        self._rotor_data.update_input_reference(applied_controls)
        
        # Debug: log rotor velocities being applied
        # ==========================================
        # 2. UNIVERSAL MOTOR RECORDING LOGIC
        # ==========================================
        # Detect ARM transition (Disarmed -> Armed)
        if currently_armed and not self._motor_was_armed:
            self._live_motor_buffer.clear()
            carb.log_warn("Started recording Motor data with absolute timestamps.")

        # Collect data while ARMED
        if currently_armed:
            if len(applied_controls) >= 4:
                self._live_motor_buffer.append({
                    "time_ns": time.time_ns(),
                    "motor0": applied_controls[0],
                    "motor1": applied_controls[1],
                    "motor2": applied_controls[2],
                    "motor3": applied_controls[3]
                })

        # Detect DISARM transition (Armed -> Disarmed)
        if not currently_armed and self._motor_was_armed:
            self._save_motor_buffer_to_csv()

        # Save state for the next tick
        self._motor_was_armed = currently_armed

    def _save_motor_buffer_to_csv(self):
        """Writes the recorded motor buffer to a CSV file upon disarm."""
        if not self._live_motor_buffer:
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"/home/geekscode/ros2_ws/src/my_drone_pkg/my_drone_pkg/live_rotors_logs/live_motor_record_{timestamp}.csv"

        try:
            with open(filename, mode='w', newline='') as csvfile:
                fieldnames = ["time_ns", "motor0", "motor1", "motor2", "motor3"]
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

                writer.writeheader()
                writer.writerows(self._live_motor_buffer)

            carb.log_warn(f"Drone DISARMED. Saved {len(self._live_motor_buffer)} motor records to {filename}")
        except Exception as e:
            carb.log_error(f"Failed to save Motor CSV: {e}")
        finally:
            self._live_motor_buffer.clear()

    def update_gps_data(self, data):
        """Gets called by the 'update_sensor' method to update the current GPS data

        Args:
            data (dict): The data produced by an GPS sensor
        """

        # GPS data
        self._sensor_data.fix_type = int(data["fix_type"])
        self._sensor_data.latitude_deg = int(data["latitude"] * 10000000)
        self._sensor_data.longitude_deg = int(data["longitude"] * 10000000)
        self._sensor_data.altitude = int(data["altitude"] * 1000)
        self._sensor_data.eph = int(data["eph"])
        self._sensor_data.epv = int(data["epv"])
        self._sensor_data.velocity = int(data["speed"] * 100)
        self._sensor_data.velocity_north = int(data["velocity_north"] * 100)
        self._sensor_data.velocity_east = int(data["velocity_east"] * 100)
        self._sensor_data.velocity_down = int(data["velocity_down"] * 100)
        self._sensor_data.cog = int(data["cog"] * 100)
        self._sensor_data.satellites_visible = int(data["sattelites_visible"])

        # Signal that we have new GPS data
        self._sensor_data.new_gps_data = True

        self._sensor_data.sim_lat = data["latitude"]
        self._sensor_data.sim_lon = data["longitude"]
        self._sensor_data.sim_alt = data["altitude"]

    def update_bar_data(self, data):
        """Gets called by the 'update_sensor' method to update the current Barometer data

        Args:
            data (dict): The data produced by an Barometer sensor
        """

        # Barometer data
        self._sensor_data.temperature = data["temperature"]
        self._sensor_data.abs_pressure = data["absolute_pressure"] * 100
        self._sensor_data.pressure_alt = data["pressure_altitude"]

        # Signal that we have new Barometer data
        self._sensor_data.new_bar_data = True

    def update_mag_data(self, data):
        """Gets called by the 'update_sensor' method to update the current Vision data

        Args:
            data (dict): The data produced by an Vision sensor
        """

        # Magnetometer data
        self._sensor_data.xmag = data["magnetic_field"][0] * 1000
        self._sensor_data.ymag = data["magnetic_field"][1] * 1000
        self._sensor_data.zmag = data["magnetic_field"][2] * 1000

        # Signal that we have new Magnetometer data
        self._sensor_data.new_mag_data = True

    def update_vision_data(self, data):
        """Method that 'in the future' will get called by the 'update_sensor' method to update the current Vision data
        This callback is currently not being called (TODO in a future simulator version)
        Args:
            data (dict): The data produced by an Vision sensor
        """

        # Vision or MOCAP data
        self._sensor_data.vision_x = data["x"]
        self._sensor_data.vision_y = data["y"]
        self._sensor_data.vision_z = data["z"]
        self._sensor_data.vision_roll = data["roll"]
        self._sensor_data.vision_pitch = data["pitch"]
        self._sensor_data.vision_yaw = data["yaw"]

        # Signal that we have new vision or mocap data
        self._sensor_data.new_vision_data = True

    def update_state(self, state: State):
        """
        Updates the internal SensorMsg instance with the latest simulation state,
        in units and conventions suitable for Betaflight/MSP (no MAVLink NED->FRD
        quaternion rotation or integer scaling).
        """

        # Quaternion [w, x, y, z] directly from sim
        attitude = state.get_attitude_ned_frd()
        self._sensor_data.sim_attitude[0] = attitude[3]  # w
        self._sensor_data.sim_attitude[1] = attitude[0]  # x
        self._sensor_data.sim_attitude[2] = attitude[1]  # y
        self._sensor_data.sim_attitude[3] = attitude[2]  # z

        # Angular velocity [rad/s]
        ang_vel = state.get_angular_velocity_frd()
        self._sensor_data.sim_angular_vel[0] = ang_vel[0]
        self._sensor_data.sim_angular_vel[1] = ang_vel[1]
        self._sensor_data.sim_angular_vel[2] = ang_vel[2]

        # Linear acceleration [m/s²]
        acc_vel = state.get_linear_acceleration_ned()
        self._sensor_data.sim_acceleration[0] = acc_vel[0]
        self._sensor_data.sim_acceleration[1] = acc_vel[1]
        self._sensor_data.sim_acceleration[2] = acc_vel[2]

        # Linear velocity [m/s] in NED frame
        lin_vel = state.get_linear_velocity_ned()
        self._sensor_data.sim_velocity_inertial[0] = lin_vel[0]
        self._sensor_data.sim_velocity_inertial[1] = lin_vel[1]
        self._sensor_data.sim_velocity_inertial[2] = lin_vel[2]

        # Airspeeds [m/s]
        body_vel = state.get_linear_body_velocity_ned_frd()
        self._sensor_data.sim_ind_airspeed = body_vel[0]
        self._sensor_data.sim_true_airspeed = float(np.linalg.norm(lin_vel))

        self._sensor_data.new_sim_state = True



    def input_reference(self):
        """
        Returns the latest motor command values (throttle/rotor speeds)
        received from Betaflight via MSP.
        """
        return self._rotor_data.input_reference

    def __del__(self):
        """
        Clean up the MSP connection when the backend is destroyed.
        """
        try:
            if self._connection:
                self._connection.disconnect()
                self._connection = None
        except Exception as e:
            carb.log_info(f"MSP connection was not closed: {e}")

    def start(self):
        """
        Start the Betaflight MSP backend connection.
        Opens the MSP communication interface and sets the running flag.
        """
        # if self._is_running:
        #     return
        pass
        # if self._api is None:
        #     self.re_initialize_interface()
        #
        # self._is_running = True
        # carb.log_info(f"Betaflight backend started for vehicle ID {self._vehicle_id}")
        # self._api.connect()
    def stop(self):
        """
        Stop the Betaflight MSP backend connection.
        Closes the MSP communication interface and cleans up state.
        """
        # if not self._is_running:
        #     return
        #
        # self._is_running = False
        #
        # if self._api:
        #     try:
        #         self._api.close_connections()
        #     except Exception as e:
        #         carb.log_warn(f"Error closing MSP connection: {e}")
        #     self._connection = None
        #
        # carb.log_info("Betaflight MSP backend stopped")
        pass
    def reset(self):
        """For now does nothing. Here for compatibility purposes only
        """
        return

    def re_initialize_interface(self):
        """
        Reset the Betaflight MSP interface to its initial state.
        This closes any previous connection, resets sensor data, and
        opens a new MSP communication channel using MultiWii.
        """
        # Restart the sensor data
        self._sensor_data = SensorMsg()

        # Close old connection if still open
        if self._api:
            try:
                self._api.disconnect()
            except Exception as e:
                carb.log_warn(f"Error closing old MSP connection: {e}")

        self._api.connect()

    def update(self, dt):
        if self.update_disabled:
            controls = self.current_controls
        else:
            controls = self._api.get_rotor_controls()
            self.current_controls = controls.copy()
        
        self.handle_control(controls)

        self._update_arm_state_and_reload_logic()
        self._current_utime += int(dt * 1_000_000)
        self.send_sensor_msgs_msp(self._current_utime)

    def send_sensor_msgs_msp(self, time_usec: int):
        """
        Send sensor data (IMU, Baro, Position, Velocity)
        """
        if not hasattr(self, "_sensor_data"):
            carb.log_warn("No sensor data object attached for send_sensor_msgs_msp()")
            return

        sensor = self._sensor_data  # shortcut


        # Build IMU dictionary
        imu = {
            "angular_velocity": [sensor.xgyro, sensor.ygyro, sensor.zgyro],  # rad/s
            "linear_acceleration": [sensor.xacc, sensor.yacc, sensor.zacc],  # m/s²
            "orientation": sensor.sim_attitude  # quaternion [w,x,y,z]
        }

        # INFO: LOGGING CURRENT IMU OF VEHICLE
        # if self._log_time + 1 < time.time():
        #     carb.log_warn(f"Vehicle {self._vehicle_id}: {imu}")
        #     self._log_time = time.time()

        # Build Baro dictionary
        baro = {
            "pressure": sensor.abs_pressure  # Pascals
        }

        # Position from sim_lat/lon/alt
        position = [
            sensor.sim_lat,
            sensor.sim_lon,
            sensor.sim_alt
        ]

        # Velocity (m/s) from simulation state
        velocity = sensor.sim_velocity_inertial  # [vx, vy, vz]

        magnetometer = [
            sensor.xmag,
            sensor.ymag,
            sensor.zmag
        ]

        try:
            # time_usec is in microseconds, convert to seconds float for send_state()
            self._api.send_state(time_usec / 1e6, imu, baro, position, velocity, magnetometer)
        except Exception as e:
            carb.log_warn(f"Error sending sensor msgs via MSP: {e}")

    def send_vision_msgs(self, time_usec: int):
        """
        MSP does not natively support vision/mocap messages like MAVLink.
        You can implement a custom MSP command if your Betaflight firmware supports it,
        or skip this if not applicable.
        """
        carb.log_info("Sending vision/mocap msgs (MSP) - Not implemented")

    def update_graphical_sensor(self, sensor_type: str, data):
        """Method that when implemented, should handle the receival of graphical sensor data

        Args:
            sensor_type (str): A name that describes the type of sensor (for example MonocularCamera)
            data (dict): A dictionary that contains the data produced by the sensor
        """
        pass

