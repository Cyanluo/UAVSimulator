import carb
import numpy as np
from scipy.spatial.transform import Rotation

from pegasus.simulator.logic.backends.backend import Backend
from pegasus.simulator.logic.controller.base_control_controller import BaseControlController
from pegasus.simulator.logic.controller.coax_ardupilot_mixer import CoaxArduPilotMixer
from pegasus.simulator.logic.interface.pegasus_interface import MultirotorState


class BaseControlBackend(Backend):
    """Pure Python backend for CoaxCopter base-control simulation."""

    def __init__(
        self,
        target_height=3.0,
        results_file=None,
        controller=None,
        mixer=None,
        hover_position=None,
        yaw_reference=None,
        trajectory_enabled=False,
        trajectory_delay=1.0,
        trajectory_radii=(1.0, 0.8, 0.35),
        trajectory_period=18.0,
        trajectory_ramp_time=3.0,
        trajectory_log_interval=1.0,
    ):
        super().__init__(config={})
        self.target_height = target_height
        self.controller = controller or BaseControlController(results_file=results_file)
        self.mixer = mixer or CoaxArduPilotMixer()
        self.hover_position = None if hover_position is None else np.asarray(hover_position, dtype=float)
        self.yaw_reference = yaw_reference
        self.trajectory_enabled = bool(trajectory_enabled)
        self.trajectory_delay = float(trajectory_delay)
        self.trajectory_radii = np.asarray(trajectory_radii, dtype=float)
        self.trajectory_period = float(trajectory_period)
        self.trajectory_ramp_time = float(trajectory_ramp_time)
        self.trajectory_log_interval = float(trajectory_log_interval)
        self.trajectory_origin = None
        self.trajectory_start_time = None
        self.trajectory_yaw = None
        self.trajectory_tracking_started = False
        self.next_trajectory_log_time = None
        self.hold_target = None
        self.state = None
        self.cmd = None
        self.input_ref = self._zero_input_reference()

    def take_off(self, height):
        if self.vehicle is None or self.vehicle.vehicle_state != MultirotorState.LAND:
            return False

        state = self.state if self.state is not None else self.vehicle.state
        target_z = self.target_height if self.target_height is not None else height
        current_position = np.asarray(state.position, dtype=float)
        target_position = current_position.copy()
        target_position[2] = target_z
        self.hold_target = target_position.copy()
        self._start_trajectory(self.hold_target)
        self.cmd = [
            target_position,
            np.zeros(3),
            np.zeros(3),
            self._current_yaw() if self.yaw_reference is None else self.yaw_reference,
            0.0,
        ]
        return True

    def hold(self):
        if self.state is None and self.hold_target is None:
            return
        hold_position = self.hold_target.copy() if self.hold_target is not None else np.asarray(self.state.position, dtype=float)
        trajectory_origin = np.asarray(self.state.position, dtype=float) if self.state is not None else hold_position
        self._start_trajectory(trajectory_origin)
        self.cmd = [
            hold_position,
            np.zeros(3),
            np.zeros(3),
            self._current_yaw() if self.yaw_reference is None else self.yaw_reference,
            0.0,
        ]

    def update_state(self, state):
        self.state = state
        self.controller.update_state(state)
        if self.hover_position is not None:
            self.cmd = [
                self.hover_position.copy(),
                np.zeros(3),
                np.zeros(3),
                self._current_yaw() if self.yaw_reference is None else self.yaw_reference,
                0.0,
            ]

    def update_sensor(self, sensor_type: str, data):
        pass

    def update_graphical_sensor(self, sensor_type: str, data):
        pass

    def input_reference(self):
        return self.input_ref

    def update(self, dt: float):
        self._update_trajectory_reference()
        control = self.controller.update(dt, self.cmd)
        if control is None:
            return

        if len(control) == 2:
            force, torques = control
            if self.vehicle:
                self.input_ref = self.vehicle.force_and_torques_to_velocities(force, torques)
                if hasattr(self.controller, "append_actuator_statistics") and isinstance(self.input_ref, dict):
                    self.controller.append_actuator_statistics(self.input_ref)
            return

        roll, pitch, yaw, throttle = control[:4]
        if len(control) >= 6:
            fx, fy = control[4:6]
            try:
                self.input_ref = self.mixer.mix(roll, pitch, yaw, throttle, fx=fx, fy=fy)
            except TypeError:
                self.input_ref = self.mixer.mix(roll, pitch, yaw, throttle)
        else:
            self.input_ref = self.mixer.mix(roll, pitch, yaw, throttle)
        if hasattr(self.controller, "append_actuator_statistics"):
            self.controller.append_actuator_statistics(self.input_ref)
        if hasattr(self.controller, "append_allocation_statistics"):
            self.controller.append_allocation_statistics(self.mixer)

    def start(self):
        self.input_ref = self._zero_input_reference()
        carb.log_warn(f"BaseControlBackend mixer: {type(self.mixer).__name__}")
        self.controller.start()

    def stop(self):
        self.cmd = None
        self.hold_target = None
        self.trajectory_origin = None
        self.trajectory_start_time = None
        self.trajectory_yaw = None
        self.trajectory_tracking_started = False
        self.next_trajectory_log_time = None
        self.input_ref = self._zero_input_reference()
        self.controller.stop()

    def reset(self):
        self.cmd = None
        self.hold_target = None
        self.trajectory_origin = None
        self.trajectory_start_time = None
        self.trajectory_yaw = None
        self.trajectory_tracking_started = False
        self.next_trajectory_log_time = None
        self.input_ref = self._zero_input_reference()
        self.controller.reset()

    def _sim_time(self):
        if self.vehicle is None:
            return 0.0
        return float(self.vehicle.pg.world.current_time)

    def _start_trajectory(self, origin):
        if not self.trajectory_enabled:
            return
        self.trajectory_origin = np.asarray(origin, dtype=float).copy()
        self.trajectory_start_time = self._sim_time()
        self.trajectory_yaw = self._current_yaw() if self.yaw_reference is None else self.yaw_reference
        self.trajectory_tracking_started = False
        self.next_trajectory_log_time = None

    def _update_trajectory_reference(self):
        if not self.trajectory_enabled or self.trajectory_origin is None or self.trajectory_start_time is None:
            return
        if self.vehicle is None or self.vehicle.vehicle_state != MultirotorState.FLYING:
            return

        t = self._sim_time() - self.trajectory_start_time - self.trajectory_delay
        if t <= 0.0:
            self.cmd = [
                self.trajectory_origin.copy(),
                np.zeros(3),
                np.zeros(3),
                self.trajectory_yaw,
                0.0,
            ]
            return

        if not self.trajectory_tracking_started:
            self.trajectory_tracking_started = True
            self.next_trajectory_log_time = 0.0
            carb.log_warn(
                "BaseControlBackend: trajectory tracking started "
                f"at t={self._sim_time():.2f}s, origin={self.trajectory_origin.tolist()}, "
                f"radii={self.trajectory_radii.tolist()}, period={self.trajectory_period:.2f}s"
            )
        self._log_trajectory_time(t)

        p_ref, v_ref, a_ref = self._trajectory_sample(t)
        self.cmd = [
            p_ref,
            v_ref,
            a_ref,
            self.trajectory_yaw,
            0.0,
        ]

    def _log_trajectory_time(self, trajectory_time):
        if self.trajectory_log_interval <= 0.0:
            return
        if self.next_trajectory_log_time is None:
            self.next_trajectory_log_time = 0.0
        if trajectory_time + 1.0e-9 < self.next_trajectory_log_time:
            return

        carb.log_warn(
            "BaseControlBackend: trajectory running "
            f"sim_t={self._sim_time():.2f}s, traj_t={trajectory_time:.2f}s"
        )
        while self.next_trajectory_log_time <= trajectory_time + 1.0e-9:
            self.next_trajectory_log_time += self.trajectory_log_interval

    def _trajectory_sample(self, t):
        omega = 2.0 * np.pi / max(self.trajectory_period, 1e-3)
        radii = self.trajectory_radii

        freqs = np.array(
            [
                [1.00, 1.29, 1.57],
                [1.17, 0.79, 1.43],
                [0.63, 1.11, 1.49],
            ],
            dtype=float,
        )
        amps = np.array(
            [
                [0.68, 0.24, 0.06],
                [0.62, 0.26, 0.07],
                [0.58, 0.20, 0.06],
            ],
            dtype=float,
        )
        phases = np.array(
            [
                [0.0, 1.7, 4.1],
                [2.2, 0.4, 5.0],
                [1.1, 3.3, 0.8],
            ],
            dtype=float,
        )

        wt = freqs * omega * t + phases
        base = radii * np.sum(amps * np.sin(wt), axis=1)
        base_d = radii * np.sum(amps * freqs * omega * np.cos(wt), axis=1)
        base_dd = -radii * np.sum(amps * (freqs * omega) ** 2 * np.sin(wt), axis=1)

        ramp, ramp_d, ramp_dd = self._smooth_ramp(t)
        p_ref = self.trajectory_origin + ramp * base
        v_ref = ramp * base_d + ramp_d * base
        a_ref = ramp * base_dd + 2.0 * ramp_d * base_d + ramp_dd * base
        return p_ref, v_ref, a_ref

    def _smooth_ramp(self, t):
        ramp_time = max(self.trajectory_ramp_time, 1e-3)
        if t >= ramp_time:
            return 1.0, 0.0, 0.0

        u = max(0.0, t / ramp_time)
        ramp = 3.0 * u * u - 2.0 * u * u * u
        ramp_d = (6.0 * u - 6.0 * u * u) / ramp_time
        ramp_dd = (6.0 - 12.0 * u) / (ramp_time * ramp_time)
        return ramp, ramp_d, ramp_dd

    def _current_yaw(self):
        state = self.state if self.state is not None else (self.vehicle.state if self.vehicle is not None else None)
        if state is None:
            return 0.0
        return Rotation.from_quat(state.attitude).as_euler("ZYX")[0]

    @staticmethod
    def _zero_input_reference():
        return {"rotor": [0.0, 0.0], "servo": [0.0, 0.0]}
