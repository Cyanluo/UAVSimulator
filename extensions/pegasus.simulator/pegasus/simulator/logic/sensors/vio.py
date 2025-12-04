__all__ = ["VIO"]

import numpy as np

from pegasus.simulator.logic.state import State
from pegasus.simulator.logic.sensors import Sensor

class VIO(Sensor):
    """ Initialize the IMU class
        >>> {"update_rate": 100.0}
    """
    def __init__(self, config={}):
        # Initialize the Super class "object" attributes
        super().__init__(sensor_type="VIO", update_rate=config.get("update_rate", 100.0))

        self._state = {
            "x": 0.0,
            "y": 0.0,
            "z": 0.0,
            "roll": 0.0,
            "pitch": 0.0,
            "yaw": 0.0,
            "pose_covariance": np.array([0.0]*21),
            "vx": 0.0,
            "vy": 0.0,
            "vz": 0.0,
            "speed_covariance": np.array([0.0]*9)
        }

    @property
    def state(self):
        """
        (dict) The 'state' of the sensor, i.e. the data produced by the sensor at any given point in time
        """
        return self._state

    @Sensor.update_at_rate
    def update(self, state: State, dt: float):
        position = state.get_position_ned()
        self._state["x"] = position[0]
        self._state["y"] = position[1]
        self._state["z"] = position[2]

        euler_angle = state.get_euler_ned_frd()
        self._state["roll"] = euler_angle[0]
        self._state["pitch"] = euler_angle[1]
        self._state["yaw"] = euler_angle[2]
        self._state["pose_covariance"] = VIO.build_pose_covariance()

        lin_vel = state.get_linear_velocity_ned()
        self._state["vx"] = lin_vel[0]
        self._state["vy"] = lin_vel[1]
        self._state["vz"] = lin_vel[2]
        self._state["speed_covariance"] = VIO.build_speed_covariance()

        return self._state

    @staticmethod
    def build_pose_covariance(linear_accel_cov=0.01, angular_vel_cov=0.01, tracker_confidence=2):
        cov_pose = linear_accel_cov * pow(10, 3 - int(tracker_confidence))
        cov_twist = angular_vel_cov * pow(10, 1 - int(tracker_confidence))
        covariance = np.array([cov_pose, 0, 0, 0, 0, 0,
                                  cov_pose, 0, 0, 0, 0,
                                     cov_pose, 0, 0, 0,
                                       cov_twist, 0, 0,
                                          cov_twist, 0,
                                             cov_twist])
        return covariance

    @staticmethod
    def build_speed_covariance(linear_accel_cov=0.01, tracker_confidence=2):
        cov_pose = linear_accel_cov * pow(10, 3 - int(tracker_confidence))
        return np.array([
            cov_pose, 0.0, 0.0,
            0.0, cov_pose, 0.0,
            0.0, 0.0, cov_pose
        ])