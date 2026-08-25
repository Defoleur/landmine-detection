import socket
import json
import threading
import time
import cv2
import numpy as np
from pxr import UsdGeom
import carb

MAX_DGRAM = 60000

class UDPHelper:
    """
    Manages all UDP sending and receiving logic in background threads.
    """
    
    # --- CHANGE 1: Added mines_port to constructor arguments ---
    def __init__(self, ip, image_port, vel_port, tag_port, pos_port, traj_ports: list, depth_port, second_image_port, mines_port=None):
        self.ip = ip
        self.image_port = image_port
        self.velocity_port = vel_port
        self.tag_port = tag_port
        self.position_port = pos_port
        self.trajectory_ports = traj_ports
        self.mines_port = mines_port # Store the new port
        self.depth_port = depth_port
        self.second_image_port = second_image_port
        
        # Socket for sending all data
        self.send_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        
        self.traj_sockets = []

        # State for received trajectory (must be thread-safe)
        self._traj_lock = threading.Lock()
        self.received_positions = []
        self.yaws = []
        self.received_data = {}

    def start_trajectory_listener(self):
        """Starts a listener loop in a daemon thread for each trajectory port."""
        carb.log_info(f"Starting UDP listeners for ports: {self.trajectory_ports}")
        
        for port in self.trajectory_ports:
            try:
                # Create and bind a new socket for each port
                sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                sock.bind((self.ip, port))
                self.traj_sockets.append(sock)
                
                # Start a dedicated thread for this socket
                thread = threading.Thread(
                    target=self._get_trajectory_loop, 
                    args=(sock, port), # Pass the socket and port to the loop
                    daemon=True
                )
                thread.start()
            except Exception as e:
                carb.log_error(f"Failed to bind trajectory socket on port {port}: {e}")
   
    def _get_trajectory_loop(self, sock, port):
        """[Internal] The blocking loop that listens for trajectory data on one port."""
        while True:
            try:
                data, _ = sock.recvfrom(65535)
                msg_list = json.loads(data.decode("utf-8"))
                
                new_points = []
                new_yaws = []

                if isinstance(msg_list, list):
                    for pos in msg_list:
                        new_points.append((pos["x"] / 100, pos["y"] / 100, pos["z"] / 100))
                        new_yaws.append(pos["yaw"])
                else:
                    new_points.append((msg_list["x"] / 100, msg_list["y"] / 100, msg_list["z"] / 100))
                    new_yaws.append(msg_list["yaw"])
                
                with self._traj_lock:
                    self.received_data[port] = (new_points, new_yaws)
                    
            except Exception as e:
                carb.log_error(f"UDP trajectory parse error on port {port}: {e}")
                # Don't sleep on error, just try to recover
            
            # (Sleep is removed to be more responsive, as recvfrom is blocking)

    def get_all_trajectories(self):
        """Safely gets the dictionary of all received trajectory data."""
        with self._traj_lock:
            # Return a copy to avoid thread issues
            return self.received_data.copy()
        
    def _send_data_thread(self, data, port):
        """[Internal] Generic sender thread for simple messages."""
        try:
            self.send_socket.sendto(data, (self.ip, port))
        except Exception as e:
            carb.log_warn(f"UDP send error to port {port}: {e}")

    def send_tag_pos(self, tag_prim):
        """Calculates and sends the tag position in a background thread."""
        if tag_prim is None:
            return

        try:
            tag_xform = UsdGeom.Xformable(tag_prim).ComputeLocalToWorldTransform(0.0)
            position = tag_xform.ExtractTranslation()
            data = {"x": position[0], "y": position[1], "z": position[2]}
            msg = json.dumps(data).encode("utf-8")
            
            # Start the sender thread
            threading.Thread(target=self._send_data_thread, args=(msg, self.tag_port), daemon=True).start()
        except Exception as e:
            carb.log_warn(f"Failed to prepare tag pos: {e}")

    def send_drone_position(self, body_prim, velo_reader):
        """
        Calculates drone pos, sends it in a background thread,
        and returns the calculated position.
        """
        if body_prim is None or velo_reader is None:
            return None

        try:
            drone_xform = UsdGeom.Xformable(body_prim).ComputeLocalToWorldTransform(0.0)
            position = drone_xform.ExtractTranslation()
            rotation_quat = drone_xform.ExtractRotation().GetQuat()
            
            qw = rotation_quat.GetReal()
            qx, qy, qz = rotation_quat.GetImaginary()
            
            velocity = velo_reader.get_velocity()
            if velocity is None:
                velocity = np.array([0.0, 0.0, 0.0])

            data = {
                "x": position[0], "y": position[1], "z": position[2],
                "qw": qw, "qx": qx, "qy": qy, "qz": qz,
                "vx": float(velocity[0]), "vy": float(velocity[1]), "vz": float(velocity[2])
            }
            msg = json.dumps(data).encode("utf-8")
            
            # Start the sender thread
            threading.Thread(target=self._send_data_thread, args=(msg, self.position_port), daemon=True).start()
            
            # Return the calculated position for the main thread (e.g., for drawing)
            return position
            
        except Exception as e:
            carb.log_warn(f"Failed to prepare drone pos: {e}")
            return None
        
    def send_all_drone_positions(self, drone_prims_dict: dict, velo_readers_dict: dict):
        """
        Calculates and sends positions for all drones in one message.
        This is a "fire-and-forget" broadcast.
        """
        if not drone_prims_dict or not velo_readers_dict:
            return

        all_drone_data = []
        all_drone_positions = []

        all_drone_positions_map = {}

        try:
            for drone_id, body_prim in drone_prims_dict.items():
                
                velo_reader = velo_readers_dict.get(drone_id)
                
                if body_prim is None or velo_reader is None:
                    carb.log_warn(f"Missing prim or reader for drone_id {drone_id}. Skipping.")
                    continue
                
                drone_xform = UsdGeom.Xformable(body_prim).ComputeLocalToWorldTransform(0.0)
                position = drone_xform.ExtractTranslation()
                all_drone_positions.append(position)
                rotation_quat = drone_xform.ExtractRotation().GetQuat()
                
                qw = rotation_quat.GetReal()
                qx, qy, qz = rotation_quat.GetImaginary()

                all_drone_positions_map[drone_id] = position
                
                velocity = velo_reader.get_velocity()
                if velocity is None:
                    velocity = np.array([0.0, 0.0, 0.0])

                data = {
                    "id": drone_id, # Add the ID
                    "x": position[0], "y": position[1], "z": position[2],
                    "qw": qw, "qx": qx, "qy": qy, "qz": qz,
                    "vx": float(velocity[0]), "vy": float(velocity[1]), "vz": float(velocity[2])
                }
                
                all_drone_data.append(data)
            
            if not all_drone_data:
                return # No valid drones found

            # Wrap all drone data in a single "drones" list
            final_payload = all_drone_data
            msg = json.dumps(final_payload).encode("utf-8")
            
            # Start one sender thread for the single message
            threading.Thread(target=self._send_data_thread, args=(msg, self.position_port), daemon=True).start()
            
            return all_drone_positions_map
        except Exception as e:
            carb.log_warn(f"Failed to prepare all drone positions: {e}")

    def _send_image_thread(self, jpg_bytes):
        """[Internal] Sender thread for chunked image data."""
        try:
            total_len = len(jpg_bytes)
            num_chunks = (total_len + MAX_DGRAM - 1) // MAX_DGRAM

            for i in range(num_chunks):
                start = i * MAX_DGRAM
                end = min(start + MAX_DGRAM, total_len)
                chunk = jpg_bytes[start:end]
                header = i.to_bytes(2, 'big') + num_chunks.to_bytes(2, 'big')
                self.send_socket.sendto(header + chunk, (self.ip, self.image_port))
        except Exception as e:
            carb.log_warn(f"UDP image send error: {e}")

    def send_image(self, img_bgr):
        # Capture the exact moment in nanoseconds
        frame_timestamp_ns = time.time_ns()
        
        def sender():
            # Encode as JPEG
            _, buffer = cv2.imencode('.jpg', img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            jpg_bytes = buffer.tobytes()

            # Split bytes into chunks
            total_len = len(jpg_bytes)
            num_chunks = (total_len + MAX_DGRAM - 1) // MAX_DGRAM
            
            # Convert the 64-bit integer timestamp to 8 bytes
            timestamp_bytes = frame_timestamp_ns.to_bytes(8, 'big')

            for i in range(num_chunks):
                start = i * MAX_DGRAM
                end = min(start + MAX_DGRAM, total_len)
                chunk = jpg_bytes[start:end]
                
                # Prefix with chunk index (2), total chunks (2), and timestamp (8). Total 12 bytes header.
                header = i.to_bytes(2, 'big') + num_chunks.to_bytes(2, 'big') + timestamp_bytes
                try:
                    self.send_socket.sendto(header + chunk, (self.ip, self.image_port))
                    self.send_socket.sendto(header + chunk, (self.ip, self.second_image_port))
                except Exception as e:
                    print(f"Send error: {e}")

        threading.Thread(target=sender, daemon=True).start()

    def send_depth_image(self, depth_array):
        """
        Encodes a float32 depth array (meters) to a 16-bit PNG (millimeters)
        and sends it over UDP in chunks using a background thread.
        """
        if depth_array is None or not hasattr(self, 'depth_port') or self.depth_port is None:
            return

        # Capture the exact moment in nanoseconds
        frame_timestamp_ns = time.time_ns()

        def sender():
            try:
                # 1. Convert meters (float32) to millimeters (uint16)
                depth_mm = np.clip(depth_array * 1000.0, 0, 65535).astype(np.uint16)

                # 2. Encode as lossless PNG 
                success, buffer = cv2.imencode('.png', depth_mm, [int(cv2.IMWRITE_PNG_COMPRESSION), 3])
                if not success:
                    carb.log_warn("Failed to encode depth image to PNG.")
                    return
                    
                png_bytes = buffer.tobytes()

                # 3. Chunk and Send
                total_len = len(png_bytes)
                num_chunks = (total_len + MAX_DGRAM - 1) // MAX_DGRAM
                
                # Convert the 64-bit integer timestamp to 8 bytes
                timestamp_bytes = frame_timestamp_ns.to_bytes(8, 'big')

                for i in range(num_chunks):
                    start = i * MAX_DGRAM
                    end = min(start + MAX_DGRAM, total_len)
                    chunk = png_bytes[start:end]
                    
                    # Prefix with chunk index (2), total chunks (2), and timestamp (8). Total 12 bytes header.
                    header = i.to_bytes(2, 'big') + num_chunks.to_bytes(2, 'big') + timestamp_bytes
                    self.send_socket.sendto(header + chunk, (self.ip, self.depth_port))
                    
            except Exception as e:
                carb.log_warn(f"UDP depth send error: {e}")

        threading.Thread(target=sender, daemon=True).start()

    def shutdown(self):
        """Closes the sockets."""
        carb.log_info("Shutting down UDP sockets.")
        self.send_socket.close()

    # --- CHANGE 2: New Function for Mines ---
    def send_mines_data(self, mines_list):
        """
        Sends the list of visible mines over UDP in a background thread.
        
        Args:
            mines_list (list[dict]): List of dicts containing mine data 
                                     (path, distance, bbox, etc.)
        """
        if mines_list is None or self.mines_port is None:
            return

        try:
            # Prepare data payload
            # You can wrap it in a dict if you want to add metadata like timestamp
            payload = {
                "count": len(mines_list),
                "mines": mines_list
            }
            
            # Serialize
            msg = json.dumps(payload).encode("utf-8")
            
            # Check size limit (UDP limit is ~65k bytes)
            # If mines_list is huge, we might need to chunk it, 
            # but for <100 mines simple sending is fine.
            if len(msg) > MAX_DGRAM:
                carb.log_warn("Mines data too large for single UDP packet! Dropping frame.")
                return

            # Send in background thread
            threading.Thread(
                target=self._send_data_thread, 
                args=(msg, self.mines_port), 
                daemon=True
            ).start()
            
        except Exception as e:
            carb.log_warn(f"Failed to send mines data: {e}")
