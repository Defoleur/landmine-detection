from omni.isaac.dynamic_control import _dynamic_control
import numpy as np
import carb

class VelocityReader:
    """
    A simple wrapper to get rigid body velocity using dynamic_control
    without conflicting with other physics interfaces.
    """
    def __init__(self, articulation_prim_path: str, body_name: str):
        self._dc = _dynamic_control.acquire_dynamic_control_interface()
        self._art_path = articulation_prim_path
        self._body_name = body_name
        self._art_handle = _dynamic_control.INVALID_HANDLE
        self._body_handle = _dynamic_control.INVALID_HANDLE
        self._default_velocity = np.array([0.0, 0.0, 0.0])

    def _get_handles(self):
        """Attempts to get the articulation and body handles."""
        if self._art_handle == _dynamic_control.INVALID_HANDLE:
            self._art_handle = self._dc.get_articulation(self._art_path)
        
        if self._art_handle != _dynamic_control.INVALID_HANDLE:
            if self._body_handle == _dynamic_control.INVALID_HANDLE:
                self._body_handle = self._dc.get_rigid_body(self._body_name)

    def get_velocity(self) -> np.ndarray:
        """
        Gets the current linear velocity.
        Returns [0,0,0] if the simulation isn't ready.
        """
        # Try to get handles if we don't have them
        if self._body_handle == _dynamic_control.INVALID_HANDLE:
            self._get_handles()
            # If still invalid after trying, sim isn't ready
            if self._body_handle == _dynamic_control.INVALID_HANDLE:
                return self._default_velocity

        # Get the physics state
        try:
            velocity_vec3 = self._dc.get_rigid_body_linear_velocity(self._body_handle)
            if velocity_vec3:
                # Convert from carb.Vec3 to numpy array
                return np.array([velocity_vec3.x, velocity_vec3.y, velocity_vec3.z])
        except Exception as e: 
            # This can happen if the sim is stopping
            carb.log_warn(e)
            pass

        return self._default_velocity

    def reset(self):
        """Resets the handles, e.g., when simulation stops."""
        self._art_handle = _dynamic_control.INVALID_HANDLE
        self._body_handle = _dynamic_control.INVALID_HANDLE