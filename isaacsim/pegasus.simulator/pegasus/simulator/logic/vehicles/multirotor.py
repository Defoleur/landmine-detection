"""
| File: multirotor.py
| Author: Marcelo Jacinto (marcelo.jacinto@tecnico.ulisboa.pt)
| License: BSD-3-Clause. Copyright (c) 2024, Marcelo Jacinto. All rights reserved.
| Description: Definition of the Multirotor class which is used as the base for all the multirotor vehicles.
"""

import numpy as np

from omni.isaac.dynamic_control import _dynamic_control
import omni.isaac.sensor as sensor
import omni.isaac.core.utils.prims as prim_utils
from pxr import PhysxSchema
# The vehicle interface
from pegasus.simulator.logic.vehicles.vehicle import Vehicle

# Mavlink interface
from pegasus.simulator.logic.backends.px4_mavlink_backend import PX4MavlinkBackend, PX4MavlinkBackendConfig

# Sensors and dynamics setup
from pegasus.simulator.logic.dynamics import LinearDrag
from pegasus.simulator.logic.thrusters import QuadraticThrustCurve
from pegasus.simulator.logic.sensors import Barometer, IMU, Magnetometer, GPS
import carb
import csv
from datetime import datetime

import threading
import json
import socket
import time

class MultirotorConfig:
    """
    A data class that is used for configuring a Multirotor
    """

    def __init__(self):
        """
        Initialization of the MultirotorConfig class
        """

        # Stage prefix of the vehicle when spawning in the world
        self.stage_prefix = "quadrotor"

        # The USD file that describes the visual aspect of the vehicle (and some properties such as mass and moments of inertia)
        self.usd_file = ""

        # The default thrust curve for a quadrotor and dynamics relating to drag
        self.thrust_curve = QuadraticThrustCurve()
        # self.drag = LinearDrag([0.50, 0.30, 0.0])
        # TODO: DRAG TEST
        # self.drag = LinearDrag([1.0, 0.60, 0.0]) # even worse
        # self.drag = LinearDrag([0.25, 0.15, 0.0]) # better
        # self.drag = LinearDrag([0.0, 0.0, 0.0]) # better
        # INFO: CURRENT VERSION
        self.drag = LinearDrag([0.1, 0.1, 0.0]) # better

        # self.drag = LinearDrag([0.25, 0.25, 0.5]) # better

        # The default sensors for a quadrotor
        # self.sensors = [Barometer(), IMU(), Magnetometer(), GPS()]
        self.sensors = [Barometer(), IMU(), Magnetometer(), GPS()]

        # The default graphical sensors for a quadrotor
        self.graphical_sensors = []

        # The default omnigraphs for a quadrotor
        self.graphs = []

        # The backends for actually sending commands to the vehicle. By default use mavlink (with default mavlink configurations)
        # [Can be None as well, if we do not desired to use PX4 with this simulated vehicle]. It can also be a ROS2 backend
        # or your own custom Backend implementation!
        self.backends = [PX4MavlinkBackend(config=PX4MavlinkBackendConfig())]


class Multirotor(Vehicle):
    """Multirotor class - It defines a base interface for creating a multirotor
    """
    def __init__(
        self,
        # Simulation specific configurations
        stage_prefix: str = "quadrotor",
        usd_file: str = "",
        vehicle_id: int = 0,
        # Spawning pose of the vehicle
        init_pos=[0.0, 0.0, 0.07],
        init_orientation=[0.0, 0.0, 0.0, 1.0],
        config=MultirotorConfig(),
    ):
        """Initializes the multirotor object

        Args:
            stage_prefix (str): The name the vehicle will present in the simulator when spawned. Defaults to "quadrotor".
            usd_file (str): The USD file that describes the looks and shape of the vehicle. Defaults to "".
            vehicle_id (int): The id to be used for the vehicle. Defaults to 0.
            init_pos (list): The initial position of the vehicle in the inertial frame (in ENU convention). Defaults to [0.0, 0.0, 0.07].
            init_orientation (list): The initial orientation of the vehicle in quaternion [qx, qy, qz, qw]. Defaults to [0.0, 0.0, 0.0, 1.0].
            config (MultirotorConfig, optional): Defaults to MultirotorConfig().
        """

        # 1. Initiate the Vehicle object itself
        super().__init__(stage_prefix, usd_file, init_pos, init_orientation, config.sensors, config.graphical_sensors, config.graphs, config.backends)

        # self._stage_prefix = stage_prefix
        self._imu_stage_prefix = stage_prefix
        # 2. Setup the dynamics of the system - get the thrust curve of the vehicle from the configuration
        self._thrusters = config.thrust_curve
        self._drag = config.drag
        self.init_pos = init_pos
        self.init_orientation = init_orientation

        self.contact_sensor = sensor.ContactSensor(
            prim_path=f"{self.prim_path}/body/contact_sensor", # A unique path for the sensor
            min_threshold=0,       # Min impulse to report
            max_threshold=1000000, # Max impulse to report
            radius=0.2,            # Optional: a sphere to approximate sensor shape for efficiency
            translation=(0,0,0)    # Position relative to its parent (the chassis)
        )

        chassis_prim_path = f"{self.prim_path}/body"
        chassis_prim = prim_utils.get_prim_at_path(chassis_prim_path)

        # Check if the prim exists and is valid
        if chassis_prim.IsValid():
            # Apply the Contact Report API to the chassis prim
            # This is the key step that prevents the error
            PhysxSchema.PhysxContactReportAPI.Apply(chassis_prim)
            carb.log_warn(f"Successfully applied Contact Report API to {chassis_prim_path}")
        # else:
        #     carb.log_warn(f"{chassis_prim_path} not found!")

        self._forces_was_armed = False
        self._live_forces_buffer = []

    def _save_forces_buffer_to_csv(self):
        """Writes the recorded aerodynamic forces buffer to a CSV file upon disarm."""
        if not self._live_forces_buffer:
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"/home/geekscode/ros2_ws/src/my_drone_pkg/my_drone_pkg/live_forces_logs/live_forces_record_{timestamp}.csv"

        try:
            with open(filename, mode='w', newline='') as csvfile:
                fieldnames = ["time_ns", "force_z_m0", "force_z_m1", "force_z_m2", "force_z_m3", "rolling_moment", "drag_x", "drag_y", "drag_z"]
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

                writer.writeheader()
                writer.writerows(self._live_forces_buffer)

            carb.log_warn(f"Forces saved. Dumped {len(self._live_forces_buffer)} records to {filename}")
        except Exception as e:
            carb.log_error(f"Failed to save Forces CSV: {e}")
        finally:
            self._live_forces_buffer.clear()

    def send_imu_biases_loop(self):
        # Find the IMU sensor once before entering the loop
        imu_sensor = None
        for sensor in self._sensors:
            if "IMU" in sensor.__class__.__name__:
                imu_sensor = sensor
                break
        
        if imu_sensor is None:
            carb.log_warn("No IMU sensor found. Canceling UDP bias stream.")
            return

        target_ip = "127.0.0.1" 
        target_port = 5016
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        carb.log_info(f"Starting continuous IMU bias stream to {target_ip}:{target_port}")

        # 3. The continuous loop
        while self._running_imu_stream:
            try:
                # Extract the drifting biases dynamically every iteration
                gyro_bias = getattr(imu_sensor, "_gyroscope_bias", [0.0, 0.0, 0.0])
                accel_bias = getattr(imu_sensor, "_accelerometer_bias", [0.0, 0.0, 0.0])

                if hasattr(gyro_bias, "tolist"):
                    gyro_bias = gyro_bias.tolist()
                if hasattr(accel_bias, "tolist"):
                    accel_bias = accel_bias.tolist()

                payload = {
                    "drone_id": self._imu_stage_prefix.split("/")[-1],
                    "gyroscope_bias": gyro_bias,
                    "accelerometer_bias": accel_bias
                }

                # carb.log_warn(payload)
                
                msg = json.dumps(payload).encode("utf-8")
                sock.sendto(msg, (target_ip, target_port))
                
            except Exception as e:
                carb.log_warn(f"Failed to send IMU biases over UDP: {e}")
            
            # Sleep to limit the send rate (e.g., 10 Hz is usually plenty for slowly drifting biases)
            time.sleep(0.02) 
        
        # Clean up when the loop finishes
        sock.close()
        carb.log_info("IMU bias stream stopped.")

    def start(self):
        """In this case we do not need to do anything extra when the simulation starts"""
        self.contact_sensor.initialize()
        carb.log_warn("start from multirotor")
        self._running_imu_stream = True
        # --- NEW: Extract IMU Biases and Send via UDP ---
        if self._imu_stage_prefix.split("/")[-1] == "quadrotor":
        # Start the daemon thread so it doesn't block the simulation startup
            threading.Thread(target=self.send_imu_biases_loop, daemon=True).start()

    def stop(self):
        """In this case we do not need to do anything extra when the simulation stops"""
        self._running_imu_stream = False
        pass

    def update(self, dt: float):
        """
        Method that computes and applies the forces to the vehicle in simulation based on the motor speed.
        This method must be implemented by a class that inherits this type. This callback
        is called on every physics step.

        Args:
            dt (float): The time elapsed between the previous and current function calls (s).
        """

        # Get the articulation root of the vehicle

        articulation = self.get_dc_interface().get_articulation(self._stage_prefix)

        # Get the desired angular velocities for each rotor from the first backend (can be mavlink or other) expressed in rad/s
        if len(self._backends) != 0:
            desired_rotor_velocities = self._backends[0].input_reference()
        else:
            desired_rotor_velocities = [0.0 for i in range(self._thrusters._num_rotors)]

        # Input the desired rotor velocities in the thruster model
        self._thrusters.set_input_reference(desired_rotor_velocities)

        # Get the desired forces to apply to the vehicle
        forces_z, _, rolling_moment = self._thrusters.update(self._state, dt)

        # Apply force to each rotor
        for i in range(4):

            # Apply the force in Z on the rotor frame
            self.apply_force([0.0, 0.0, forces_z[i]], body_part="/rotor" + str(i))

            # Generate the rotating propeller visual effect
            self.handle_propeller_visual(i, forces_z[i], articulation)

        # Apply the torque to the body frame of the vehicle that corresponds to the rolling moment
        self.apply_torque([0.0, 0.0, rolling_moment], "/body")

        # Compute the total linear drag force to apply to the vehicle's body frame
        drag = self._drag.update(self._state, dt)
        self.apply_force(drag, body_part="/body")

        # carb.log_warn(f"[multirotor] applying forces for motors: {desired_rotor_velocities}")

        # Call the update methods in all backends
        for backend in self._backends:
            backend.update(dt)
        
        # ==========================================
        # --- FORCES RECORDING LOGIC ---
        # ==========================================
        
        # We determine if the drone is "Armed" by checking if any rotor has velocity
        currently_armed = any(v > 100 for v in desired_rotor_velocities)

        # Detect ARM transition
        if currently_armed and not self._forces_was_armed:
            self._live_forces_buffer.clear()
            carb.log_warn("Started recording dynamic aerodynamic forces.")

        # Collect data while ARMED
        if currently_armed:
            self._live_forces_buffer.append({
                "time_ns": time.time_ns(),
                "force_z_m0": forces_z[0],
                "force_z_m1": forces_z[1],
                "force_z_m2": forces_z[2],
                "force_z_m3": forces_z[3],
                "rolling_moment": rolling_moment,
                "drag_x": drag[0], 
                "drag_y": drag[1], 
                "drag_z": drag[2],
            })
            # carb.log_warn(f"Saving forces row. Desired rotor velocities: {desired_rotor_velocities}")

        # Detect DISARM transition
        if not currently_armed and self._forces_was_armed:
            self._save_forces_buffer_to_csv()

        self._forces_was_armed = currently_armed

    def handle_propeller_visual(self, rotor_number, force: float, articulation):
        """
        Auxiliar method used to set the joint velocity of each rotor (for animation purposes) based on the
        amount of force being applied on each joint

        Args:
            rotor_number (int): The number of the rotor to generate the rotation animation
            force (float): The force that is being applied on that rotor
            articulation (_type_): The articulation group the joints of the rotors belong to
        """

        # Rotate the joint to yield the visual of a rotor spinning (for animation purposes only)
        joint = self.get_dc_interface().find_articulation_dof(articulation, "joint" + str(rotor_number))

        # Spinning when armed but not applying force
        if 0.0 < force < 0.1:
            self.get_dc_interface().set_dof_velocity(joint, 5 * self._thrusters.rot_dir[rotor_number])
        # Spinning when armed and applying force
        elif 0.1 <= force:
            self.get_dc_interface().set_dof_velocity(joint, 100 * self._thrusters.rot_dir[rotor_number])
        # Not spinning
        else:
            self.get_dc_interface().set_dof_velocity(joint, 0)

    def force_and_torques_to_velocities(self, force: float, torque: np.ndarray):
        """
        Auxiliar method used to get the target angular velocities for each rotor, given the total desired thrust [N] and
        torque [Nm] to be applied in the multirotor's body frame.

        Note: This method assumes a quadratic thrust curve. This method will be improved in a future update,
        and a general thrust allocation scheme will be adopted. For now, it is made to work with multirotors directly.

        Args:
            force (np.ndarray): A vector of the force to be applied in the body frame of the vehicle [N]
            torque (np.ndarray): A vector of the torque to be applied in the body frame of the vehicle [Nm]

        Returns:
            list: A list of angular velocities [rad/s] to apply in reach rotor to accomplish suchs forces and torques
        """

        # Get the body frame of the vehicle
        rb = self.get_dc_interface().get_rigid_body(self._stage_prefix + "/body")

        # Get the rotors of the vehicle
        rotors = [self.get_dc_interface().get_rigid_body(self._stage_prefix + "/rotor" + str(i)) for i in range(self._thrusters._num_rotors)]

        # Get the relative position of the rotors with respect to the body frame of the vehicle (ignoring the orientation for now)
        relative_poses = self.get_dc_interface().get_relative_body_poses(rb, rotors)

        # Define the alocation matrix
        aloc_matrix = np.zeros((4, self._thrusters._num_rotors))

        # Define the first line of the matrix (T [N])
        aloc_matrix[0, :] = np.array(self._thrusters._rotor_constant)

        # Define the second and third lines of the matrix (\tau_x [Nm] and \tau_y [Nm])
        aloc_matrix[1, :] = np.array([relative_poses[i].p[1] * self._thrusters._rotor_constant[i] for i in range(self._thrusters._num_rotors)])
        aloc_matrix[2, :] = np.array([-relative_poses[i].p[0] * self._thrusters._rotor_constant[i] for i in range(self._thrusters._num_rotors)])

        # Define the forth line of the matrix (\tau_z [Nm])
        aloc_matrix[3, :] = np.array([self._thrusters._rolling_moment_coefficient[i] * self._thrusters._rot_dir[i] for i in range(self._thrusters._num_rotors)])

        # Compute the inverse allocation matrix, so that we can get the angular velocities (squared) from the total thrust and torques
        aloc_inv = np.linalg.pinv(aloc_matrix)

        # Compute the target angular velocities (squared)
        squared_ang_vel = aloc_inv @ np.array([force, torque[0], torque[1], torque[2]])

        # Making sure that there is no negative value on the target squared angular velocities
        squared_ang_vel[squared_ang_vel < 0] = 0.0

        # ------------------------------------------------------------------------------------------------
        # Saturate the inputs while preserving their relation to each other, by performing a normalization
        # ------------------------------------------------------------------------------------------------
        max_thrust_vel_squared = np.power(self._thrusters.max_rotor_velocity[0], 2)
        max_val = np.max(squared_ang_vel)

        if max_val >= max_thrust_vel_squared:
            normalize = np.maximum(max_val / max_thrust_vel_squared, 1.0)

            squared_ang_vel = squared_ang_vel / normalize

        # Compute the angular velocities for each rotor in [rad/s]
        ang_vel = np.sqrt(squared_ang_vel)

        return ang_vel

    def destroy(self):
        self._running_imu_stream = False
        super().destroy()

    def stop_backends(self):
        super().stop_backends()

    def start_backends(self):
        super().start_backends()

    def check_for_collisions(self):
        """Checks the sensor and prints info about any collisions."""
        # Get the latest sensor readings
        readings = self.contact_sensor.get_current_frame()
        
        if 'in_contact' in readings:
            return readings['in_contact']
        
        return False
