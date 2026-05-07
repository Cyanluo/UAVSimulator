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
        takeoff_climb_rate=0.6,
        takeoff_z_leash=0.3,
        takeoff_delay=0.0,
        takeoff_throttle_slew_time=2.0,
        takeoff_spool_release_throttle=None,
        takeoff_spool_min_altitude=0.12,
        trajectory_enabled=False,
        trajectory_delay=1.0,
        trajectory_radii=(1.0, 0.8, 0.35),
        trajectory_period=18.0,
        trajectory_ramp_time=3.0,
    ):
        super().__init__(config={})
        self.target_height = target_height
        self.controller = controller or BaseControlController(results_file=results_file)
        self.mixer = mixer or CoaxArduPilotMixer()
        self.hover_position = None if hover_position is None else np.asarray(hover_position, dtype=float)
        self.yaw_reference = yaw_reference
        self.takeoff_climb_rate = float(takeoff_climb_rate)
        self.takeoff_z_leash = float(takeoff_z_leash)
        self.takeoff_delay = float(takeoff_delay)
        self.takeoff_throttle_slew_time = float(takeoff_throttle_slew_time)
        self.takeoff_spool_release_throttle = takeoff_spool_release_throttle
        self.takeoff_spool_min_altitude = float(takeoff_spool_min_altitude)
        self.trajectory_enabled = bool(trajectory_enabled)
        self.trajectory_delay = float(trajectory_delay)
        self.trajectory_radii = np.asarray(trajectory_radii, dtype=float)
        self.trajectory_period = float(trajectory_period)
        self.trajectory_ramp_time = float(trajectory_ramp_time)
        self.trajectory_origin = None
        self.trajectory_start_time = None
        self.trajectory_yaw = None
        self.trajectory_tracking_started = False
        self.takeoff_target = None
        self.takeoff_spool_active = False
        self.takeoff_spool_throttle = 0.0
        self.hold_target = None
        self.state = None
        self.cmd = None
        self.input_ref = self._zero_input_reference()

    def take_off(self, height):
        if self.vehicle is None or self.vehicle.vehicle_state != MultirotorState.LAND:
            return False
        if self._sim_time() < self.takeoff_delay:
            return False

        state = self.state if self.state is not None else self.vehicle.state
        target_z = self.target_height if self.target_height is not None else height
        current_position = np.asarray(state.position, dtype=float)
        self.takeoff_target = current_position.copy()
        self.takeoff_target[2] = target_z
        self.hold_target = self.takeoff_target.copy()
        self.cmd = [
            current_position.copy(),
            np.zeros(3),
            np.zeros(3),
            self._current_yaw() if self.yaw_reference is None else self.yaw_reference,
            0.0,
        ]
        self.takeoff_spool_active = True
        self.takeoff_spool_throttle = 0.0
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
        if self.takeoff_spool_active:
            self._update_takeoff_spool(dt)
            return

        self._update_takeoff_reference(dt)
        self._update_trajectory_reference()
        control = self.controller.update(dt, self.cmd)
        if control is None:
            return

        if len(control) == 2:
            force, torques = control
            if self.vehicle:
                self.input_ref = self.vehicle.force_and_torques_to_velocities(force, torques)
            return

        roll, pitch, yaw, throttle = control
        self.input_ref = self.mixer.mix(roll, pitch, yaw, throttle)

    def start(self):
        self.input_ref = self._zero_input_reference()
        self.controller.start()

    def stop(self):
        self.cmd = None
        self.takeoff_target = None
        self.takeoff_spool_active = False
        self.takeoff_spool_throttle = 0.0
        self.hold_target = None
        self.trajectory_origin = None
        self.trajectory_start_time = None
        self.trajectory_yaw = None
        self.trajectory_tracking_started = False
        self.input_ref = self._zero_input_reference()
        self.controller.stop()

    def reset(self):
        self.cmd = None
        self.takeoff_target = None
        self.takeoff_spool_active = False
        self.takeoff_spool_throttle = 0.0
        self.hold_target = None
        self.trajectory_origin = None
        self.trajectory_start_time = None
        self.trajectory_yaw = None
        self.trajectory_tracking_started = False
        self.input_ref = self._zero_input_reference()
        self.controller.reset()

    def _sim_time(self):
        if self.vehicle is None:
            return 0.0
        return float(self.vehicle.pg.world.current_time)

    def _update_takeoff_reference(self, dt):
        if self.takeoff_target is None or self.cmd is None or dt <= 0.0:
            return

        current_target = self.cmd[0].copy()
        state_z = self.state.position[2] if self.state is not None else current_target[2]
        remaining = self.takeoff_target[2] - current_target[2]
        step = self.takeoff_climb_rate * dt
        if abs(remaining) <= step:
            current_target[2] = self.takeoff_target[2]
            self.takeoff_target = None
        else:
            desired_z = current_target[2] + self.takeoff_climb_rate * np.sign(remaining) * dt
            leash_z = state_z + self.takeoff_z_leash
            current_target[2] = min(desired_z, leash_z, self.takeoff_target[2])

        self.cmd[0] = current_target
        self.cmd[1] = np.zeros(3)

    def _update_takeoff_spool(self, dt):
        if dt <= 0.0:
            return

        slew_time = max(self.takeoff_throttle_slew_time, 1e-3)
        self.takeoff_spool_throttle = min(1.0, self.takeoff_spool_throttle + dt / slew_time)
        self.input_ref = self.mixer.mix(0.0, 0.0, 0.0, self.takeoff_spool_throttle)

        release_throttle = self.takeoff_spool_release_throttle
        if release_throttle is None:
            release_throttle = getattr(self.controller, "hover_percentage", 0.48) + 0.03

        altitude = self.state.position[2] if self.state is not None else 0.0
        if self.takeoff_spool_throttle < release_throttle and altitude < self.takeoff_spool_min_altitude:
            return

        self.takeoff_spool_active = False
        self.controller.start()
        if hasattr(self.controller, "filtered_throttle"):
            self.controller.filtered_throttle = float(self.takeoff_spool_throttle)
        if self.state is not None and self.cmd is not None:
            current_position = np.asarray(self.state.position, dtype=float)
            self.cmd[0] = current_position.copy()
            if self.takeoff_target is not None and self.takeoff_target[2] > current_position[2]:
                self.cmd[0][2] = min(current_position[2] + self.takeoff_z_leash, self.takeoff_target[2])
            self.cmd[1] = np.zeros(3)

    def _start_trajectory(self, origin):
        if not self.trajectory_enabled:
            return
        self.trajectory_origin = np.asarray(origin, dtype=float).copy()
        self.trajectory_start_time = self._sim_time()
        self.trajectory_yaw = self._current_yaw() if self.yaw_reference is None else self.yaw_reference
        self.trajectory_tracking_started = False

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
            carb.log_warn(
                "BaseControlBackend: trajectory tracking started "
                f"at t={self._sim_time():.2f}s, origin={self.trajectory_origin.tolist()}, "
                f"radii={self.trajectory_radii.tolist()}, period={self.trajectory_period:.2f}s"
            )

        p_ref, v_ref, a_ref = self._trajectory_sample(t)
        self.cmd = [
            p_ref,
            v_ref,
            a_ref,
            self.trajectory_yaw,
            0.0,
        ]

    def _trajectory_sample(self, t):
        omega = 2.0 * np.pi / max(self.trajectory_period, 1e-3)
        phase = np.array([omega * t, 0.5 * omega * t, 0.75 * omega * t])
        radii = self.trajectory_radii

        base = radii * np.sin(phase)
        base_d = radii * np.array([omega, 0.5 * omega, 0.75 * omega]) * np.cos(phase)
        base_dd = -radii * np.array([omega, 0.5 * omega, 0.75 * omega]) ** 2 * np.sin(phase)

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
