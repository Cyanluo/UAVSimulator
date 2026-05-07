import numpy as np
from scipy.spatial.transform import Rotation

from pegasus.simulator.logic.backends.backend import Backend
from pegasus.simulator.logic.controller.coaxcopter_position_controller import trajController
from pegasus.simulator.logic.interface.pegasus_interface import MultirotorState


class LegacyCoaxPythonBackend(Backend):
    """Pure Python backend that mirrors the validated ROS1 coax control path."""

    def __init__(self, target_height=3.0, results_file=None):
        super().__init__(config={})
        self.target_height = target_height
        self.controller = trajController(results_file)
        self.state = None
        self.cmd = None
        self.input_ref = self._zero_input_reference()

    def take_off(self, height):
        if self.vehicle is None or self.vehicle.vehicle_state != MultirotorState.LAND:
            return False

        state = self._state_or_vehicle_state()
        if state is None:
            return False

        target_z = self.target_height if self.target_height is not None else height
        self.cmd = [
            np.asarray([state.position[0], state.position[1], target_z], dtype=float),
            np.asarray([0.0, 0.0, 0.5], dtype=float),
            np.zeros(3),
            np.zeros(3),
            0.0,
            0.1,
        ]
        return True

    def hold(self):
        state = self._state_or_vehicle_state()
        if state is None:
            return

        self.cmd = [
            np.asarray([state.position[0], state.position[1], state.position[2]], dtype=float),
            np.zeros(3),
            np.zeros(3),
            np.zeros(3),
            Rotation.from_quat(state.attitude).as_euler("ZYX")[0],
            0.0,
        ]

    def update_state(self, state):
        self.state = state
        self.controller.update_state(state)

    def update_sensor(self, sensor_type: str, data):
        pass

    def update_graphical_sensor(self, sensor_type: str, data):
        pass

    def input_reference(self):
        return self.input_ref

    def update(self, dt: float):
        force, torques = self.controller.update(dt, self.cmd)
        if self.vehicle:
            self.input_ref = self.vehicle.force_and_torques_to_velocities(force, torques)

    def start(self):
        self.input_ref = self._zero_input_reference()
        self.controller.start()

    def stop(self):
        self.cmd = None
        self.input_ref = self._zero_input_reference()
        self.controller.stop()

    def reset(self):
        self.cmd = None
        self.input_ref = self._zero_input_reference()
        self.controller.reset()

    def _state_or_vehicle_state(self):
        if self.state is not None:
            return self.state
        if self.vehicle is not None:
            return self.vehicle.state
        return None

    @staticmethod
    def _zero_input_reference():
        return {"rotor": [0.0, 0.0], "servo": [0.0, 0.0]}
