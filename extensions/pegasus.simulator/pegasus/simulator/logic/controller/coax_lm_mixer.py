import numpy as np


class CoaxLMMixer:
    """Levenberg-Marquardt control allocation based on the thesis model.

    The controller currently provides normalized roll/pitch/yaw/throttle
    commands. This allocator maps them to a pseudo body wrench, then solves the
    nonlinear inverse allocation for rotor speeds and servo angles.
    """

    def __init__(
        self,
        max_rotor_velocity=900.0,
        min_rotor_velocity=0.0,
        servo_limit_rad=1.413,
        throttle_hover=0.21,
        throttle_thrust_max=1.0,
        force_coefficient=None,
        moment_coefficient=None,
        arm_upper=0.24,
        arm_lower=0.24,
        roll_moment_scale=0.13,
        pitch_moment_scale=0.13,
        yaw_moment_scale=0.055,
        yaw_sign=-1.0,
        wrench_weights=(0.05, 0.05, 3.0, 1.0, 1.0, 2.0),
        max_iterations=12,
        lambda_initial=1.0e-3,
        lambda_factor=5.0,
        tolerance=1.0e-7,
    ):
        self.max_rotor_velocity = float(max_rotor_velocity)
        self.min_rotor_velocity = float(min_rotor_velocity)
        self.servo_limit_rad = float(servo_limit_rad)
        self.throttle_hover = float(throttle_hover)
        self.throttle_thrust_max = float(throttle_thrust_max)

        # Default to normalized thrust: Cf * omega_max^2 = 1.
        self.force_coefficient = (
            float(force_coefficient)
            if force_coefficient is not None
            else 1.0 / (self.max_rotor_velocity * self.max_rotor_velocity)
        )
        self.moment_coefficient = (
            float(moment_coefficient)
            if moment_coefficient is not None
            else 0.0408 * self.force_coefficient
        )
        self.arm_upper = float(arm_upper)
        self.arm_lower = float(arm_lower)
        self.roll_moment_scale = float(roll_moment_scale)
        self.pitch_moment_scale = float(pitch_moment_scale)
        self.yaw_moment_scale = float(yaw_moment_scale)
        self.yaw_sign = float(yaw_sign)
        self.wrench_weights = np.asarray(wrench_weights, dtype=float)
        self.max_iterations = int(max_iterations)
        self.lambda_initial = float(lambda_initial)
        self.lambda_factor = float(lambda_factor)
        self.tolerance = float(tolerance)

        self._last_solution = None
        self.limits = {}
        self.last_target_wrench = np.full(6, np.nan)
        self.last_allocated_wrench = np.full(6, np.nan)
        self.last_residual = np.full(6, np.nan)
        self.last_raw_residual_norm = np.nan
        self.last_weighted_residual_norm = np.nan

    @classmethod
    def from_airframe(
        cls,
        *,
        max_rotor_velocity=900.0,
        min_rotor_velocity=0.0,
        servo_limit_rad=1.413,
        throttle_hover=0.21,
        throttle_thrust_max=1.0,
        force_coefficient=None,
        yaw_torque_ratio=0.0408,
        moment_coefficient=None,
        arm_upper=0.24,
        arm_lower=0.24,
        roll_authority=0.45,
        pitch_authority=0.45,
        yaw_authority=0.65,
        yaw_sign=-1.0,
        wrench_weights=None,
        **kwargs,
    ):
        """Build a mixer from airframe parameters plus normalized authority.

        The LM solver uses physical-ish wrench units, while the controller
        sends normalized [-1, 1] attitude commands. The authority values say
        what fraction of the nominal hover actuator authority should correspond
        to a full normalized roll/pitch/yaw command.
        """
        max_rotor_velocity = float(max_rotor_velocity)
        throttle_hover = float(throttle_hover)
        force_coefficient = (
            float(force_coefficient)
            if force_coefficient is not None
            else 1.0 / (max_rotor_velocity * max_rotor_velocity)
        )
        moment_coefficient = (
            float(moment_coefficient)
            if moment_coefficient is not None
            else float(yaw_torque_ratio) * force_coefficient
        )

        hover_rotor_thrust = max(throttle_hover, 1e-6)
        servo_authority = np.sin(float(servo_limit_rad))
        roll_moment_scale = abs(float(arm_lower)) * hover_rotor_thrust * servo_authority * float(roll_authority)
        pitch_moment_scale = abs(float(arm_upper)) * hover_rotor_thrust * servo_authority * float(pitch_authority)
        yaw_moment_scale = 2.0 * abs(moment_coefficient / force_coefficient) * hover_rotor_thrust * float(yaw_authority)

        if wrench_weights is None:
            wrench_weights = cls.default_wrench_weights(
                force_coefficient=force_coefficient,
                moment_coefficient=moment_coefficient,
                arm_upper=arm_upper,
                arm_lower=arm_lower,
            )

        return cls(
            max_rotor_velocity=max_rotor_velocity,
            min_rotor_velocity=min_rotor_velocity,
            servo_limit_rad=servo_limit_rad,
            throttle_hover=throttle_hover,
            throttle_thrust_max=throttle_thrust_max,
            force_coefficient=force_coefficient,
            moment_coefficient=moment_coefficient,
            arm_upper=arm_upper,
            arm_lower=arm_lower,
            roll_moment_scale=roll_moment_scale,
            pitch_moment_scale=pitch_moment_scale,
            yaw_moment_scale=yaw_moment_scale,
            yaw_sign=yaw_sign,
            wrench_weights=wrench_weights,
            **kwargs,
        )

    @staticmethod
    def default_wrench_weights(
        force_coefficient=None,
        moment_coefficient=None,
        arm_upper=0.24,
        arm_lower=0.24,
        force_z_weight=3.0,
        moment_weight=1.0,
        yaw_weight=2.0,
        lateral_force_weight=0.05,
    ):
        """Return solver weights grouped by physical role.

        The absolute coefficient values are less important than the priority:
        keep collective thrust first, attitude moments next, and let lateral
        force be soft because servos inevitably create it while producing
        roll/pitch moments.
        """
        _ = force_coefficient, moment_coefficient, arm_upper, arm_lower
        return (
            float(lateral_force_weight),
            float(lateral_force_weight),
            float(force_z_weight),
            float(moment_weight),
            float(moment_weight),
            float(yaw_weight),
        )

    def mix(self, roll, pitch, yaw, throttle, fx=0.0, fy=0.0):
        roll = self._clip(roll, -1.0, 1.0)
        pitch = self._clip(pitch, -1.0, 1.0)
        yaw = self._clip(yaw, -1.0, 1.0)
        throttle = self._clip(throttle, 0.0, self.throttle_thrust_max)
        fx = self._clip(fx, -1.0, 1.0)
        fy = self._clip(fy, -1.0, 1.0)

        self.limits = {
            "roll": abs(roll) >= 1.0,
            "pitch": abs(pitch) >= 1.0,
            "yaw": abs(yaw) >= 1.0,
            "throttle_lower": throttle <= 0.0,
            "throttle_upper": throttle >= self.throttle_thrust_max,
            "force_x": abs(fx) >= 1.0,
            "force_y": abs(fy) >= 1.0,
        }

        y_des = self._desired_wrench(roll, pitch, yaw, throttle, fx, fy)
        u0 = self._initial_guess(roll, pitch, yaw, throttle, y_des)
        solution = self._solve_lm(u0, y_des)
        self._last_solution = solution
        self.last_target_wrench = y_des.copy()
        self.last_allocated_wrench = self._forward(solution)
        self.last_residual = self.last_allocated_wrench - self.last_target_wrench
        self.last_raw_residual_norm = float(np.linalg.norm(self.last_residual))
        self.last_weighted_residual_norm = float(np.linalg.norm(self.last_residual * self.wrench_weights))

        omega_u, omega_d, delta_u, delta_d = solution
        return {
            "rotor": [float(omega_u), float(omega_d)],
            "servo": [float(delta_u), float(delta_d)],
        }

    def _desired_wrench(self, roll, pitch, yaw, throttle, fx=0.0, fy=0.0):
        # Paper convention has negative Fz for upward rotor thrust.
        fz = -2.0 * throttle

        # Positive roll/pitch are mapped to body moments. With the thesis model,
        # positive Mx/My produce the same servo signs as the validated APM mixer.
        mx = self.roll_moment_scale * roll
        my = self.pitch_moment_scale * pitch
        mz = self.yaw_sign * self.yaw_moment_scale * yaw
        return np.array([fx, fy, fz, mx, my, mz], dtype=float)

    def _initial_guess(self, roll, pitch, yaw, throttle, y_des):
        up_thrust = self._clip(throttle + self.yaw_sign * 0.5 * yaw, 0.0, 1.0)
        down_thrust = self._clip(throttle - self.yaw_sign * 0.5 * yaw, 0.0, 1.0)
        omega_u = self._thrust_to_omega(up_thrust)
        omega_d = self._thrust_to_omega(down_thrust)

        actuator_den = self._clip(max(self.throttle_hover * 0.5, throttle), 0.5, 1.0)
        delta_u = self._clip(-pitch / actuator_den, -1.0, 1.0) * self.servo_limit_rad
        delta_d = self._clip(-roll / actuator_den, -1.0, 1.0) * self.servo_limit_rad
        analytic = np.array([omega_u, omega_d, delta_u, delta_d], dtype=float)

        if self._last_solution is None:
            return analytic

        if np.linalg.norm(self._weighted_error(self._last_solution, y_des)) < np.linalg.norm(self._weighted_error(analytic, y_des)):
            return self._last_solution.copy()
        return analytic

    def _solve_lm(self, u, y_des):
        u = self._project(u)
        damping = self.lambda_initial

        for _ in range(self.max_iterations):
            error = self._weighted_error(u, y_des)
            if np.linalg.norm(error) < self.tolerance:
                break

            jac = self._jacobian(u) * self.wrench_weights[:, None]
            hessian = jac.T @ jac
            gradient = jac.T @ error
            system = hessian + damping * np.eye(4)

            try:
                step = np.linalg.solve(system, -gradient)
            except np.linalg.LinAlgError:
                step = -np.linalg.pinv(system) @ gradient

            candidate = self._project(u + step)
            if np.linalg.norm(self._weighted_error(candidate, y_des)) < np.linalg.norm(error):
                u = candidate
                damping = max(damping / self.lambda_factor, 1.0e-9)
            else:
                damping = min(damping * self.lambda_factor, 1.0e6)

        return u

    def _weighted_error(self, u, y_des):
        return (self._forward(u) - y_des) * self.wrench_weights

    def _forward(self, u):
        omega_u, omega_d, delta_u, delta_d = u
        cf = self.force_coefficient
        cm = self.moment_coefficient
        lu = self.arm_upper
        ld = self.arm_lower
        wu2 = omega_u * omega_u
        wd2 = omega_d * omega_d
        su = np.sin(delta_u)
        sd = np.sin(delta_d)
        cu = np.cos(delta_u)
        cd = np.cos(delta_d)

        return np.array(
            [
                cf * wu2 * su,
                cf * wd2 * sd,
                -cf * wu2 * cu - cf * wd2 * cd,
                -cm * wu2 * su - ld * cf * wd2 * sd,
                cm * wd2 * sd - lu * cf * wu2 * su,
                cm * wu2 * cu - cm * wd2 * cd,
            ],
            dtype=float,
        )

    def _jacobian(self, u):
        omega_u, omega_d, delta_u, delta_d = u
        cf = self.force_coefficient
        cm = self.moment_coefficient
        lu = self.arm_upper
        ld = self.arm_lower
        wu2 = omega_u * omega_u
        wd2 = omega_d * omega_d
        su = np.sin(delta_u)
        sd = np.sin(delta_d)
        cu = np.cos(delta_u)
        cd = np.cos(delta_d)

        return np.array(
            [
                [2.0 * cf * omega_u * su, 0.0, cf * wu2 * cu, 0.0],
                [0.0, 2.0 * cf * omega_d * sd, 0.0, cf * wd2 * cd],
                [-2.0 * cf * omega_u * cu, -2.0 * cf * omega_d * cd, cf * wu2 * su, cf * wd2 * sd],
                [-2.0 * cm * omega_u * su, -2.0 * ld * cf * omega_d * sd, -cm * wu2 * cu, -ld * cf * wd2 * cd],
                [-2.0 * lu * cf * omega_u * su, 2.0 * cm * omega_d * sd, -lu * cf * wu2 * cu, cm * wd2 * cd],
                [2.0 * cm * omega_u * cu, -2.0 * cm * omega_d * cd, -cm * wu2 * su, cm * wd2 * sd],
            ],
            dtype=float,
        )

    def _project(self, u):
        u = np.asarray(u, dtype=float).copy()
        u[0] = self._clip(u[0], self.min_rotor_velocity, self.max_rotor_velocity)
        u[1] = self._clip(u[1], self.min_rotor_velocity, self.max_rotor_velocity)
        u[2] = self._clip(u[2], -self.servo_limit_rad, self.servo_limit_rad)
        u[3] = self._clip(u[3], -self.servo_limit_rad, self.servo_limit_rad)
        return u

    def _thrust_to_omega(self, thrust):
        thrust = self._clip(thrust, 0.0, 1.0)
        return np.sqrt(thrust / self.force_coefficient) if self.force_coefficient > 0.0 else 0.0

    @staticmethod
    def _clip(value, lower, upper):
        return float(np.clip(value, lower, upper))
