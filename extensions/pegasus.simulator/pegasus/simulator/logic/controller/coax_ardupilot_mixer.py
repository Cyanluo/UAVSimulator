import numpy as np


class CoaxArduPilotMixer:
    """ArduPilot-style engineering mixer for the current two-servo coax model.

    This is intended as a temporary, easy-to-debug allocator while the base
    control law is brought up. It accepts normalized roll/pitch/yaw/throttle
    commands and returns the Pegasus CoaxCopter drive dictionary.

    Sign convention follows the existing validated CoaxCopter mixer:
    - positive yaw lowers upper rotor thrust and raises lower rotor thrust
    - positive pitch maps to negative upper servo angle
    - positive roll maps to negative lower servo angle
    """

    def __init__(
        self,
        max_rotor_velocity=900.0,
        servo_limit_rad=1.413,
        servo_command_range_rad=1.57,
        servo_torque_norm=0.6,
        throttle_hover=0.48,
        throttle_thrust_max=1.0,
        throttle_avg_max=0.6,
        yaw_headroom=200.0,
        yaw_sign=-1.0,
        roll_servo_sign=-1.0,
        pitch_servo_sign=-1.0,
        min_rotor_thrust=0.0,
        max_rotor_thrust_delta=None,
        thrust_to_velocity="sqrt",
    ):
        self.max_rotor_velocity = max_rotor_velocity
        self.servo_limit_rad = servo_limit_rad
        self.servo_command_range_rad = servo_command_range_rad
        self.servo_torque_norm = servo_torque_norm
        self.throttle_hover = throttle_hover
        self.throttle_thrust_max = throttle_thrust_max
        self.throttle_avg_max = throttle_avg_max
        self.yaw_headroom = yaw_headroom
        self.yaw_sign = yaw_sign
        self.roll_servo_sign = roll_servo_sign
        self.pitch_servo_sign = pitch_servo_sign
        self.min_rotor_thrust = min_rotor_thrust
        self.max_rotor_thrust_delta = max_rotor_thrust_delta
        self.thrust_to_velocity = thrust_to_velocity
        self.limits = {}

    def mix(self, roll, pitch, yaw, throttle):
        roll_thrust = self._clip(roll, -1.0, 1.0)
        pitch_thrust = self._clip(pitch, -1.0, 1.0)
        yaw_thrust = self._clip(yaw, -1.0, 1.0)
        throttle_thrust = self._clip(throttle, 0.0, self.throttle_thrust_max)
        throttle_avg_max = self._clip(self.throttle_avg_max, throttle_thrust, self.throttle_thrust_max)

        self.limits = {
            "roll": False,
            "pitch": False,
            "yaw": False,
            "throttle_lower": throttle <= 0.0,
            "throttle_upper": throttle >= self.throttle_thrust_max,
        }

        rp_thrust_max = max(abs(roll_thrust), abs(pitch_thrust))
        if np.isclose(rp_thrust_max, 0.0):
            rp_scale = 1.0
        else:
            yaw_reserved = min(abs(yaw_thrust), 0.5 * self.yaw_headroom * 0.001)
            rp_scale = self._clip((1.0 - yaw_reserved) / rp_thrust_max, 0.0, 1.0)
            if rp_scale < 1.0:
                self.limits["roll"] = True
                self.limits["pitch"] = True

        roll_thrust *= rp_scale
        pitch_thrust *= rp_scale

        actuator_allowed = 2.0 * (1.0 - rp_scale * rp_thrust_max)
        if abs(yaw_thrust) > actuator_allowed:
            yaw_thrust = self._clip(yaw_thrust, -actuator_allowed, actuator_allowed)
            self.limits["yaw"] = True

        thrust_min_rpy = max(abs(rp_thrust_max * rp_scale), abs(yaw_thrust))
        thr_adj = throttle_thrust - throttle_avg_max
        if thr_adj < thrust_min_rpy - throttle_avg_max:
            thr_adj = min(thrust_min_rpy, throttle_avg_max) - throttle_avg_max

        thrust_out = self._clip(throttle_avg_max + thr_adj, 0.0, self.throttle_thrust_max)
        if abs(yaw_thrust) > thrust_out:
            yaw_thrust = self._clip(yaw_thrust, -thrust_out, thrust_out)
            self.limits["yaw"] = True

        # Match CoaxCopter.force_and_torques_to_velocities():
        # up = force - 0.5 * yaw, down = force + 0.5 * yaw.
        up_thrust = self._clip(thrust_out + self.yaw_sign * 0.5 * yaw_thrust, self.min_rotor_thrust, 1.0)
        down_thrust = self._clip(thrust_out - self.yaw_sign * 0.5 * yaw_thrust, self.min_rotor_thrust, 1.0)
        if self.max_rotor_thrust_delta is not None:
            up_thrust, down_thrust = self._adjust_pair(up_thrust, down_thrust, self.max_rotor_thrust_delta)

        thrust_out_actuator = self._clip(max(self.throttle_hover * 0.5, thrust_out), 0.5, 1.0)
        roll_actuator = self._clip(roll_thrust / thrust_out_actuator, -1.0, 1.0) * self.servo_limit_rad
        pitch_actuator = self._clip(pitch_thrust / thrust_out_actuator, -1.0, 1.0) * self.servo_limit_rad

        return {
            "rotor": [
                self._normalized_thrust_to_velocity(up_thrust),
                self._normalized_thrust_to_velocity(down_thrust),
            ],
            "servo": [
                # Match the validated mapping:
                # upper servo = -pitch command, lower servo = -roll command.
                self.pitch_servo_sign * pitch_actuator,
                self.roll_servo_sign * roll_actuator,
            ],
        }

    def _normalized_thrust_to_velocity(self, value):
        value = self._clip(value, 0.0, 1.0)
        if self.thrust_to_velocity == "sqrt":
            return np.sqrt(value) * self.max_rotor_velocity
        return value * self.max_rotor_velocity

    def _legacy_servo_command(self, command, thrust):
        angle = command / self.servo_torque_norm * self.servo_command_range_rad
        angle = self._decay_scalar(angle, thrust)
        return self._clip(angle, -self.servo_limit_rad, self.servo_limit_rad)

    @staticmethod
    def _adjust_pair(a, b, dmax):
        swapped = False
        if a < b:
            a, b = b, a
            swapped = True

        diff = a - b
        if diff <= dmax:
            return (b, a) if swapped else (a, b)

        delta = diff - dmax
        a_new = a - delta / 2.0
        b_new = b + delta / 2.0
        return (b_new, a_new) if swapped else (a_new, b_new)

    @staticmethod
    def _decay_scalar(a, b, delta=0.2, b0=0.6, k=0.5, gamma=1.6):
        b = max(0.0, min(1.0, b))
        diff = b - b0
        scale = 1.0 - k * (np.sign(diff) if diff != 0 else 0.0) * (abs(diff) ** gamma)
        y_cand = a * scale
        return max(a - delta, min(a + delta, y_cand))

    @staticmethod
    def _clip(value, low, high):
        return float(np.clip(value, low, high))
