import socket
import struct
import threading
import time
from collections import namedtuple
from typing import List, Optional
import carb

# Simple container types (mimic C++ struct layout)
FdmPacket = namedtuple(
    "FdmPacket",
    [
        "timestamp",
        "imu_angular_velocity_rpy",  # 3 doubles
        "imu_linear_acceleration_xyz",  # 3 doubles
        "imu_orientation_quat",  # 4 doubles (w,x,y,z)
        "velocity_xyz",  # 3 doubles
        "position_xyz",  # 3 doubles
        "pressure",  # double
    ],
)

class BetaflightApiPython:
    """
    Python translation of the C++ BetaflightApi.
    - Sends FDM state as a packed struct to state_port (UDP)
    - Receives a ServoPacket (motor PWM floats) from motor_port (UDP)
    """

    # defaults (same as C++ example)
    kBetaflightMotorCount = 4
    kBetaflightServoCount = 8

    def __init__(
        self,
        udp_ip: str = "127.0.0.1",
        servo_port: int = 9001,
        motor_port: int = 9002,
        state_port: int = 9003,
        rc_port: int = 9004,
        local_bind: str = "0.0.0.0",
    ):
        self.ip = udp_ip
        self.servo_port = servo_port
        self.motor_port = motor_port
        self.state_port = state_port
        self.rc_port = rc_port
        self.local_bind = local_bind

        # rotor and servo controls
        self.rotor_controls = [0.0] * self.kBetaflightMotorCount
        self.servo_controls = [0.0] * self.kBetaflightServoCount

        # sockets
        self._motor_sock: Optional[socket.socket] = None  # receives servo/motor data
        self._state_sock: Optional[socket.socket] = None  # send state
        self._recv_thread: Optional[threading.Thread] = None
        self._recv_thread_stop = threading.Event()

        # connection state
        self.is_connected = False
        self.failed_pkts = 0
        self.max_failed_pkts = 5
        self.start_timestamp = 0.0
        self.wait_time = 5.0

        # timing
        self.start_time = time.time()
        self._last_state_send = 0.0

        # lock for rotor_controls
        self._lock = threading.Lock()

        self.received_count = 0

        self.received_controls = []

        self.last_received_controls = []

        self._new_motor_event = threading.Event()

        # initialize sockets (not bound until connect())
        self.connect()

    def close_connections(self):
        try:
            if self._recv_thread:
                self._recv_thread_stop.set()
                self._recv_thread.join(timeout=0.5)
        except Exception:
            pass

        try:
            if self._motor_sock:
                self._motor_sock.close()
                self._motor_sock = None
        except Exception:
            pass

        try:
            if self._state_sock:
                self._state_sock.close()
                self._state_sock = None
        except Exception:
            pass

    def connect(self):
        """Create and bind UDP sockets."""
        self.close_connections()

        # motor socket: receive servo/motor tuples from Betaflight SITL
        self._motor_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._motor_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._motor_sock.bind((self.local_bind, self.motor_port))
        self._motor_sock.settimeout(0.1)

        # state socket: for sending state to Betaflight (target ip + state_port)
        self._state_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        # note: we do not bind the state socket; we'll sendto (ip, state_port)

        # start receiver thread
        self._recv_thread_stop.clear()
        self._recv_thread = threading.Thread(target=self._recv_loop, daemon=True)
        self._recv_thread.start()

        print(f"[BetaflightApi] bound motor socket on {self.local_bind}:{self.motor_port}, sending state to {self.ip}:{self.state_port}")

    def reset_rotors(self):
        with self._lock:
            for i in range(self.kBetaflightMotorCount):
                self.rotor_controls[i] = 0.0

    def _recv_loop(self):
        """Continuously attempt to receive ServoPacket-sized datagrams and update rotor_controls."""
        # ServoPacket in C++ was: float pwm[kBetaflightMotorCount]
        fmt = "<" + ("f" * self.kBetaflightMotorCount)  # little-endian floats
        size = struct.calcsize(fmt)
        start_time = time.time()
        self.received_count = 0
        self.received_controls = []

        while not self._recv_thread_stop.is_set():
            try:
                data, addr = self._motor_sock.recvfrom(4096)
                if len(data) >= size:
                    # parse only the first kBetaflightMotorCount floats
                    vals = struct.unpack_from(fmt, data, 0)
                    with self._lock:
                        for i in range(min(self.kBetaflightMotorCount, len(vals))):
                            self.rotor_controls[i] = float(vals[i])
                        self.received_count += 1
                        self.received_controls.append(self.rotor_controls.copy())
                        self._new_motor_event.set()
                    self.last_received_controls.append(self.rotor_controls.copy())
                    if len(self.last_received_controls) >= 10:
                        self.last_received_controls = []
                    self.failed_pkts = 0
                    self.is_connected = True
                    end_time = time.time()
                       
                    # if end_time - start_time >= 1:
                    #     start_time = end_time
                    #     # carb.log_warn(f"Count: {count}")
                    #     count = 0
                    # debug:
                    # print("[BetaflightApi] pwm received:", self.rotor_controls)
                else:
                    # partial or wrong size; ignore
                    self.failed_pkts += 1
                    # small sleep to avoid busy loop
                    # time.sleep(0.001)
            except socket.timeout:
                pass
                # no data this iteration
                #time.sleep(0.001)
            except Exception as e:
                print("[BetaflightApi] recv error:", e)
                time.sleep(0.01)

    def send_state(self, timestamp: float, imu: dict, baro: dict, position=None, velocity=None, magnetometer=None):
        """
        Send FdmPacket to the state_port.
        imu: dict with keys:
            - angular_velocity: [x,y,z] (radians/sec)
            - linear_acceleration: [x,y,z] (m/s^2)
            - orientation: quaternion [w,x,y,z]
        baro: dict with keys:
            - pressure
        position, velocity: optional lists [x,y,z]
        """
        if self._state_sock is None:
            return

        # Build packet fields (double precision)
        timestamp_d = float(timestamp)
        ang = imu.get("angular_velocity", [0.0, 0.0, 0.0])
        lin_acc = imu.get("linear_acceleration", [0.0, 0.0, 0.0])
        quat = imu.get("orientation", [1.0, 0.0, 0.0, 0.0])
        pos = position or [0.0, 0.0, 0.0]
        vel = velocity or [0.0, 0.0, 0.0]
        pressure = float(baro.get("pressure", 101325.0))
        mag = magnetometer or [0.0, 0.0, 0.0]

        # Pack as C struct of doubles; follow the C++ field order:
        # timestamp (double)
        # imu_angular_velocity_rpy[3] (3 doubles)
        # imu_linear_acceleration_xyz[3] (3 doubles)
        # imu_orientation_quat[4] (4 doubles)
        # velocity_xyz[3] (3 doubles)
        # position_xyz[3] (3 doubles)
        # pressure (double)
        fmt = "<" + "d" * (1 + 3 + 3 + 4 + 3 + 3 + 1 + 3)  # count of doubles
        values = (
            timestamp_d,
            ang[0], ang[1], ang[2],
            lin_acc[0], lin_acc[1], lin_acc[2],
            quat[0], quat[1], quat[2], quat[3],
            vel[0], vel[1], vel[2],
            pos[0], pos[1], pos[2],
            pressure,
            mag[0], mag[1], mag[2]
        )
        try:
            packed = struct.pack(fmt, *values)
            # carb.log_warn(f"Packed values: {packed}")
            # self._motor_sock.sendto(packed, (self.ip, self.state_port))
            self._state_sock.sendto(packed, (self.ip, self.state_port))
            self._last_state_send = time.time()
        except Exception as e:
            print("[BetaflightApi] send_state error:", e)

    def recv_control_once(self, timeout_ms: int = 10) -> bool:
        """
        Single-shot attempt to check for new motor data (event-based).
        Returns True if new packet received and rotor_controls updated.
        Note: This method now uses the event-based mechanism instead of directly
        reading from the socket (which would race with _recv_loop).
        """
        if self._motor_sock is None:
            return False

        # Wait for the event to be set by _recv_loop
        timeout_sec = timeout_ms / 1000.0
        if self._new_motor_event.wait(timeout=timeout_sec):
            # Event was set, clear it for next time
            self._new_motor_event.clear()
            self.failed_pkts = 0
            self.is_connected = True
            return True
        else:
            # Timeout - no new data
            self.failed_pkts += 1
            return False

    # Note: The update() method is now handled by BetaflightBackend.update()
    # This method is kept for backward compatibility but is no longer used
    # def update(self, dt: float, imu: dict, baro: dict, position=None, velocity=None):
    #     """
    #     Call from your simulation loop. Sends state periodically and checks/receives control.
    #     - dt: elapsed seconds
    #     - imu, baro: dictionaries as expected by send_state
    #     """
    #     timestamp = time.time() - self.start_time
    #
    #     # If we're waiting after packet failures, skip until start_timestamp
    #     if timestamp < self.start_timestamp:
    #         return
    #
    #     # Send state at up to 100 Hz (avoid flooding)
    #     now = time.time()
    #     # if (now - self._last_state_send) >= (1.0 / 60.0):  # adjustable send rate
    #     self.send_state(timestamp, imu, baro, position=position, velocity=velocity)
    #
    #     # Try to receive control (non-blocking handled in thread; this function can still poll once)
    #     got = self.recv_control_once(timeout_ms=5)
    #     # got = self._new_motor_event.wait(timeout=0.005)
    #     if got:
    #         self.failed_pkts = 0
    #         self.is_connected = True
    #     else:
    #         self.failed_pkts += 1
    #         if self.failed_pkts >= self.max_failed_pkts:
    #             self.failed_pkts = 0
    #             self.is_connected = False
    #             self.start_timestamp = timestamp + self.wait_time
    #             self.reset_rotors()
    #             print(f"[BetaflightApi] Rotors data not received. Retry in {self.wait_time} sec")

    # API-like helpers (mimic the C++ interface)
    def is_api_control_enabled(self) -> bool:
        return True

    def arm_disarm(self, arm: bool) -> bool:
        """Return if connected (we cannot truly arm here)."""
        return self.is_connected

    def get_actuation(self, rotor_index: int) -> float:
        with self._lock:
            return float(self.rotor_controls[rotor_index])

    def wait_for_new_control(self, timeout=0.01):
        """Wait for new motor data. Returns True if new data arrived."""
        return self._new_motor_event.wait(timeout=timeout)

    def get_rotor_controls(self, blocking=False, timeout=1.0):
        if blocking:
            received = self._new_motor_event.wait(timeout=timeout)
            if not received:
                carb.log_warn("[BetaflightApi] TIMEOUT: No motor data received! Sim and FC desync.")
                return None
            # Clear event for next step only after consuming new data
            self._new_motor_event.clear()
            with self._lock:
                # carb.log_warn(f"Received rotors count: {self.received_count}, values: {self.received_controls}")
                self.received_count = 0
                self.received_controls = []
                return self.rotor_controls.copy()
        else:
            # Non-blocking: just return current controls without clearing event
            with self._lock:
                return self.rotor_controls.copy()

    # def get_rotor_controls(self, blocking=True, timeout=1.0) -> []:
    #     if blocking:
    #         # Pause Isaac Sim until _recv_loop flags that new data arrived
    #         received = self._new_motor_event.wait(timeout=timeout)
                
    #         if not received:
    #             carb.log_warn("[BetaflightApi] TIMEOUT: No motor data received! Sim and FC desync.")
                    
    #     # Immediately clear the flag so the NEXT physics step has to wait for a NEW packet
    #     self._new_motor_event.clear()
        
    #     with self._lock:
    #         return self.rotor_controls

    def get_actuator_count(self) -> int:
        return self.kBetaflightMotorCount

    def set_rc_data(self, rc_data) -> bool:
        """Store RCData-like structure (opaque)."""
        self._last_rc_data = rc_data
        self.is_rc_connected = True
        return True

    def get_rc_data(self):
        """Return last rc data if supplied."""
        return getattr(self, "_last_rc_data", None)

    def start(self):
        """(re)start receiver thread if needed"""
        if self._recv_thread is None or not self._recv_thread.is_alive():
            self.connect()

    def stop(self):
        self.close_connections()

    def __del__(self):
        self.close_connections()
