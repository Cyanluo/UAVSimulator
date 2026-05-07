import os
import csv

import carb
import numpy as np
from scipy.linalg import solve_continuous_are
from scipy.spatial.transform import Rotation

from pegasus.simulator.logic.state import State


class BaseControlController:
    """Base-control law for CoaxCopter simulation.

    The controller is intentionally independent from the actuator allocator.
    It computes normalized throttle, roll, pitch and yaw demands that can be
    consumed by the temporary ArduPilot-style mixer or by the future paper
    Levenberg-Marquardt allocator.
    """

    def __init__(
        self,
        results_file=None,
        Kp=(5.0, 5.0, 7.0),
        Kv=(1.5, 1.5, 1.5),
        Kvi=(0.2, 0.2, 0.2),
        Kvd=(0.0, 0.0, 0.0),
        Kr=(53.0, 75.0, 40.0),
        Kir=(10.0, 10.0, 15.0),
        Kdr=(0.1, 0.6, 0.05),
        Kw=(0.013, 0.013, 0.01),
        Kiw=(0.002, 0.002, 0.0015),
        Kdw=(0.000008, 0.000006, 0.00015),
        attitude_control_mode="cascade_pid",
        attitude_integral_limit=(0.3, 0.3, 0.3),
        apm_xy_posvel_controller=None,
        apm_xy_pos_p=1.0,
        apm_xy_vel_p=1.2,
        apm_xy_vel_i=0.0,
        apm_xy_vel_d=0.0,
        apm_xy_vel_imax=1.0,
        apm_xy_vel_filter_hz=20.0,
        ap_angle_p=(4.05, 4.05, 4.05),
        ap_rate_p=(0.1215, 0.1215, 0.162),
        ap_rate_i=(0.1215, 0.1215, 0.0162),
        ap_rate_d=(0.00324, 0.00324, 0.0),
        ap_rate_imax=(0.45, 0.45, 0.45),
        ap_rate_filter_hz=(20.0, 20.0, 2.5),
        lqr_kRP=3.0,
        lqr_kRI=0.8,
        lqr_q=(4.0, 4.0, 2.0, 80.0, 80.0, 35.0, 4.0, 4.0, 2.0),
        lqr_ru=(1.0, 1.0, 1.0),
        max_tilt_deg=25.0,
        hover_percentage=0.48,
        mass=1.30,
        gravity=9.81,
        inertia=(0.03, 0.03, 0.05),
        torque_scale=(0.018, 0.014, 0.020),
        yaw_rate_gain=0.0,
        output_mode="normalized",
        apm_height_priority=None,
        apm_z_accel_controller=None,
        apm_z_pos_p=1.0,
        apm_z_vel_p=5.0,
        apm_z_accel_p=0.03,
        apm_z_accel_i=0.10,
        apm_z_accel_d=0.0,
        apm_z_accel_imax=0.8,
        apm_z_accel_filter_hz=10.0,
        apm_z_accel_limit=2.5,
        apm_z_use_accel_feedback=True,
        command_z_leash=None,
        throttle_filter_hz=2.0,
        takeoff_min_throttle=None,
        takeoff_altitude_gate=0.8,
        takeoff_error_gate=0.25,
        z_priority_takeoff=True,
        apply_angle_boost=True,
        angle_boost_min_cos=0.1,
    ):
        self.Kp = np.diag(Kp)
        self.Kv = np.diag(Kv)
        self.Kvi = np.diag(Kvi)
        self.Kvd = np.diag(Kvd)
        self.Kr = np.diag(Kr)
        self.Kir = np.diag(Kir)
        self.Kdr = np.diag(Kdr)
        self.Kw = np.diag(Kw)
        self.Kiw = np.diag(Kiw)
        self.Kdw = np.diag(Kdw)
        self.attitude_control_mode = attitude_control_mode
        self.attitude_integral_limit = np.asarray(attitude_integral_limit, dtype=float)
        if apm_xy_posvel_controller is None:
            apm_xy_posvel_controller = attitude_control_mode == "ardupilot_pid"
        self.apm_xy_posvel_controller = bool(apm_xy_posvel_controller)
        self.apm_xy_pos_p = float(apm_xy_pos_p)
        self.apm_xy_vel_p = float(apm_xy_vel_p)
        self.apm_xy_vel_i = float(apm_xy_vel_i)
        self.apm_xy_vel_d = float(apm_xy_vel_d)
        self.apm_xy_vel_imax = float(apm_xy_vel_imax)
        self.apm_xy_vel_filter_hz = float(apm_xy_vel_filter_hz)
        self.ap_angle_p = np.asarray(ap_angle_p, dtype=float)
        self.ap_rate_p = np.asarray(ap_rate_p, dtype=float)
        self.ap_rate_i = np.asarray(ap_rate_i, dtype=float)
        self.ap_rate_d = np.asarray(ap_rate_d, dtype=float)
        self.ap_rate_imax = np.asarray(ap_rate_imax, dtype=float)
        self.ap_rate_filter_hz = np.asarray(ap_rate_filter_hz, dtype=float)
        self.lqr_kRP = float(lqr_kRP)
        self.lqr_kRI = float(lqr_kRI)
        self.lqr_q = np.asarray(lqr_q, dtype=float)
        self.lqr_ru = np.asarray(lqr_ru, dtype=float)
        self.lqr_gain = self._build_lqr_gain()

        self.max_tilt = np.deg2rad(max_tilt_deg)
        self.hover_percentage = hover_percentage
        self.m = mass
        self.g = gravity
        self.inertia = np.diag(inertia) if np.asarray(inertia).ndim == 1 else np.asarray(inertia, dtype=float)
        self.acc2thr = self.g / self.hover_percentage
        self.torque_scale = np.asarray(torque_scale, dtype=float)
        self.yaw_rate_gain = yaw_rate_gain
        self.output_mode = output_mode
        if apm_height_priority is None:
            apm_height_priority = attitude_control_mode == "ardupilot_pid"
        self.apm_height_priority = bool(apm_height_priority)
        if apm_z_accel_controller is None:
            apm_z_accel_controller = attitude_control_mode == "ardupilot_pid"
        self.apm_z_accel_controller = bool(apm_z_accel_controller)
        self.apm_z_pos_p = float(apm_z_pos_p)
        self.apm_z_vel_p = float(apm_z_vel_p)
        self.apm_z_accel_p = float(apm_z_accel_p)
        self.apm_z_accel_i = float(apm_z_accel_i)
        self.apm_z_accel_d = float(apm_z_accel_d)
        self.apm_z_accel_imax = float(apm_z_accel_imax)
        self.apm_z_accel_filter_hz = float(apm_z_accel_filter_hz)
        self.apm_z_accel_limit = float(apm_z_accel_limit)
        self.apm_z_use_accel_feedback = bool(apm_z_use_accel_feedback)
        self.command_z_leash = None if command_z_leash is None else float(command_z_leash)
        self.throttle_filter_hz = float(throttle_filter_hz)
        if takeoff_min_throttle is None:
            takeoff_min_throttle = hover_percentage + 0.05
        self.takeoff_min_throttle = float(takeoff_min_throttle)
        self.takeoff_altitude_gate = float(takeoff_altitude_gate)
        self.takeoff_error_gate = float(takeoff_error_gate)
        self.z_priority_takeoff = bool(z_priority_takeoff)
        self.apply_angle_boost = bool(apply_angle_boost)
        self.angle_boost_min_cos = float(angle_boost_min_cos)

        self.p = np.zeros(3)
        self.v = np.zeros(3)
        self.w = np.zeros(3)
        self.a = np.zeros(3)
        self.R = Rotation.identity()
        self.received_first_state = False

        self.position_error_int = np.zeros(3)
        self.eR_int = np.zeros(3)
        self.ew_int = np.zeros(3)
        self.prev_eR = np.zeros(3)
        self.prev_ew = np.zeros(3)
        self.prev_ev = np.zeros(3)
        self.apm_xy_vel_int = np.zeros(2)
        self.apm_xy_filtered_vel_error = np.zeros(2)
        self.apm_xy_prev_filtered_vel_error = np.zeros(2)
        self.ap_rate_int = np.zeros(3)
        self.ap_prev_rate_error = np.zeros(3)
        self.ap_filtered_rate_error = np.zeros(3)
        self.ap_prev_filtered_rate_error = np.zeros(3)
        self.apm_z_accel_int = 0.0
        self.apm_z_filtered_accel_error = 0.0
        self.apm_z_prev_filtered_accel_error = 0.0
        self.filtered_throttle = self.hover_percentage
        self.total_time = 0.0

        if results_file is not None:
            self.results_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), results_file)
        else:
            self.results_file = None

        self.time_vector = []
        self.position_over_time = []
        self.desired_position_over_time = []
        self.position_error_over_time = []
        self.velocity_error_over_time = []
        self.attitude_error_over_time = []
        self.attitude_rate_error_over_time = []
        self.attitude_over_time = []
        self.desired_attitude_over_time = []
        self.angular_rate_over_time = []
        self.desired_angular_rate_over_time = []
        self.control_over_time = []
        self.last_w_des = np.zeros(3)

    def start(self):
        self.reset_statistics()
        self.position_error_int = np.zeros(3)
        self.eR_int = np.zeros(3)
        self.ew_int = np.zeros(3)
        self.prev_eR = np.zeros(3)
        self.prev_ew = np.zeros(3)
        self.prev_ev = np.zeros(3)
        self.apm_xy_vel_int = np.zeros(2)
        self.apm_xy_filtered_vel_error = np.zeros(2)
        self.apm_xy_prev_filtered_vel_error = np.zeros(2)
        self.ap_rate_int = np.zeros(3)
        self.ap_prev_rate_error = np.zeros(3)
        self.ap_filtered_rate_error = np.zeros(3)
        self.ap_prev_filtered_rate_error = np.zeros(3)
        self.apm_z_accel_int = 0.0
        self.apm_z_filtered_accel_error = 0.0
        self.apm_z_prev_filtered_accel_error = 0.0
        self.filtered_throttle = self.hover_percentage

    def stop(self):
        self.received_first_state = False
        if self.results_file is not None and len(self.time_vector) > 0:
            os.makedirs(os.path.dirname(self.results_file), exist_ok=True)
            np.savez(
                self.results_file,
                time=np.asarray(self.time_vector),
                p=np.vstack(self.position_over_time),
                desired_p=np.vstack(self.desired_position_over_time),
                ep=np.vstack(self.position_error_over_time),
                ev=np.vstack(self.velocity_error_over_time),
                er=np.vstack(self.attitude_error_over_time),
                ew=np.vstack(self.attitude_rate_error_over_time),
                attitude=np.vstack(self.attitude_over_time),
                desired_attitude=np.vstack(self.desired_attitude_over_time),
                angular_rate=np.vstack(self.angular_rate_over_time),
                desired_angular_rate=np.vstack(self.desired_angular_rate_over_time),
                control=np.vstack(self.control_over_time),
            )
            self._save_csv_statistics()
            carb.log_warn("Base control statistics saved to: " + self.results_file)
        self.reset_statistics()

    def reset(self):
        self.stop()
        self.start()

    def update_state(self, state: State):
        self.p = np.asarray(state.position, dtype=float)
        self.v = np.asarray(state.linear_velocity, dtype=float)
        self.w = np.asarray(state.angular_velocity, dtype=float)
        self.a = np.asarray(state.linear_acceleration, dtype=float)
        self.R = Rotation.from_quat(state.attitude)
        self.received_first_state = True

    def update(self, dt, cmd):
        if not self.received_first_state or cmd is None or dt <= 0.0:
            return 0.0, 0.0, 0.0, 0.0

        self.total_time += dt
        p_des, v_des, a_des, yaw_des, yaw_rate_des = self._parse_cmd(cmd)
        p_des, v_des = self._apply_command_z_leash(p_des, v_des)

        ep = self.p - p_des
        ev = self.v - v_des
        self.position_error_int += ep * dt
        ev_d = (ev - self.prev_ev) / dt
        self.prev_ev = ev

        takeoff_window = self._in_takeoff_window(p_des[2])
        near_ground_takeoff = p_des[2] >= self.p[2] and self.p[2] < self.takeoff_altitude_gate

        if self.apm_xy_posvel_controller:
            accel_xy = self._apm_xy_accel(p_des[:2], v_des[:2], a_des[:2], dt)
            xy_scale = self._takeoff_xy_scale(p_des[2])
            if self.z_priority_takeoff and xy_scale < 1.0:
                accel_xy *= xy_scale
                self.apm_xy_vel_int *= max(0.0, 1.0 - 2.0 * dt)
            elif not self.z_priority_takeoff:
                accel_xy *= xy_scale
            accel_z = -(self.Kp[2, 2] * ep[2]) - (self.Kv[2, 2] * ev[2])
            accel_z += -(self.Kvi[2, 2] * self.position_error_int[2]) - (self.Kvd[2, 2] * ev_d[2]) + a_des[2]
            accel_target = np.array([accel_xy[0], accel_xy[1], accel_z])
        else:
            accel_target = -(self.Kp @ ep) - (self.Kv @ ev) - (self.Kvi @ self.position_error_int)
            accel_target += -(self.Kvd @ ev_d) + a_des

        if self.apm_height_priority:
            attitude_accel = np.array([accel_target[0], accel_target[1], self.g])
            if self.apm_z_accel_controller:
                throttle_direct = self._apm_z_throttle(p_des[2], v_des[2], a_des[2], dt, takeoff_window)
                collective_accel = None
            else:
                throttle_direct = None
                collective_accel = float(accel_target[2] + self.g)
        else:
            throttle_direct = None
            attitude_accel = accel_target + np.array([0.0, 0.0, self.g])
            collective_accel = None

        b3_des = self._limit_tilt(attitude_accel)
        R_des = self._desired_rotation(b3_des, yaw_des)

        R = self.R.as_matrix()
        e_R = 0.5 * self.vee(R_des.T @ R - R.T @ R_des)
        self.eR_int += e_R * dt
        self.eR_int = np.clip(self.eR_int, -self.attitude_integral_limit, self.attitude_integral_limit)
        tau_cmd, e_w = self._attitude_control(R, R_des, e_R, yaw_rate_des, dt)

        if throttle_direct is not None:
            throttle = throttle_direct
            if self.apply_angle_boost:
                throttle = self._angle_boost_value(throttle, R, R_des)
            throttle = float(np.clip(throttle, 0.0, 1.0))
        elif collective_accel is None:
            collective_accel = float(np.dot(attitude_accel, R[:, 2]))
            throttle = float(np.clip(collective_accel / self.acc2thr, 0.0, 1.0))
        else:
            if self.apply_angle_boost:
                collective_accel = self._angle_boost_value(collective_accel, R, R_des)
            throttle = float(np.clip(collective_accel / self.acc2thr, 0.0, 1.0))
        throttle = self._filter_throttle(throttle, dt)
        if takeoff_window or near_ground_takeoff:
            throttle = max(throttle, self.takeoff_min_throttle)
            if self.takeoff_min_throttle > 0.0:
                self.filtered_throttle = throttle
        if self.output_mode == "force_torque":
            force = throttle
            torques = np.clip(tau_cmd, -1.0, 1.0)
            self._append_statistics(p_des, ep, ev, R, R_des, e_R, e_w, np.r_[torques, force])
            return force, torques

        if self.attitude_control_mode == "ardupilot_pid":
            roll = float(np.clip(tau_cmd[0], -1.0, 1.0))
            pitch = float(np.clip(tau_cmd[1], -1.0, 1.0))
            yaw = float(np.clip(tau_cmd[2] + self.yaw_rate_gain * yaw_rate_des, -1.0, 1.0))
        else:
            roll = float(np.clip(tau_cmd[0] * self.torque_scale[0], -1.0, 1.0))
            pitch = float(np.clip(tau_cmd[1] * self.torque_scale[1], -1.0, 1.0))
            yaw = float(np.clip(tau_cmd[2] * self.torque_scale[2] + self.yaw_rate_gain * yaw_rate_des, -1.0, 1.0))

        self._append_statistics(p_des, ep, ev, R, R_des, e_R, e_w, np.array([roll, pitch, yaw, throttle]))
        return roll, pitch, yaw, throttle

    def _append_statistics(self, p_des, ep, ev, R, R_des, e_R, e_w, control):
        self.time_vector.append(self.total_time)
        self.position_over_time.append(self.p.copy())
        self.desired_position_over_time.append(p_des.copy())
        self.position_error_over_time.append(ep.copy())
        self.velocity_error_over_time.append(ev.copy())
        self.attitude_error_over_time.append(e_R.copy())
        self.attitude_rate_error_over_time.append(e_w.copy())
        self.attitude_over_time.append(self._matrix_to_euler_xyz(R))
        self.desired_attitude_over_time.append(self._matrix_to_euler_xyz(R_des))
        self.angular_rate_over_time.append(self.w.copy())
        self.desired_angular_rate_over_time.append(self.last_w_des.copy())
        self.control_over_time.append(control.copy())

    def _save_csv_statistics(self):
        csv_file = os.path.splitext(self.results_file)[0] + ".csv"
        os.makedirs(os.path.dirname(csv_file), exist_ok=True)

        header = [
            "time",
            "p_x",
            "p_y",
            "p_z",
            "p_des_x",
            "p_des_y",
            "p_des_z",
            "ep_x",
            "ep_y",
            "ep_z",
            "ev_x",
            "ev_y",
            "ev_z",
            "att_roll",
            "att_pitch",
            "att_yaw",
            "att_des_roll",
            "att_des_pitch",
            "att_des_yaw",
            "eR_x",
            "eR_y",
            "eR_z",
            "w_x",
            "w_y",
            "w_z",
            "w_des_x",
            "w_des_y",
            "w_des_z",
            "ew_x",
            "ew_y",
            "ew_z",
            "control_0",
            "control_1",
            "control_2",
            "control_3",
        ]

        with open(csv_file, "w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(header)
            for row in zip(
                self.time_vector,
                self.position_over_time,
                self.desired_position_over_time,
                self.position_error_over_time,
                self.velocity_error_over_time,
                self.attitude_over_time,
                self.desired_attitude_over_time,
                self.attitude_error_over_time,
                self.angular_rate_over_time,
                self.desired_angular_rate_over_time,
                self.attitude_rate_error_over_time,
                self.control_over_time,
            ):
                time, p, p_des, ep, ev, attitude, desired_attitude, e_R, w, w_des, e_w, control = row
                writer.writerow(
                    [time]
                    + list(p)
                    + list(p_des)
                    + list(ep)
                    + list(ev)
                    + list(attitude)
                    + list(desired_attitude)
                    + list(e_R)
                    + list(w)
                    + list(w_des)
                    + list(e_w)
                    + list(control)
                )
        carb.log_warn("Base control CSV statistics saved to: " + csv_file)

    def _parse_cmd(self, cmd):
        if len(cmd) == 6:
            p_des, v_des, a_des, _j_des, yaw_des, yaw_rate_des = cmd
        else:
            p_des, v_des, a_des, yaw_des, yaw_rate_des = cmd
        return (
            np.asarray(p_des, dtype=float),
            np.asarray(v_des, dtype=float),
            np.asarray(a_des, dtype=float),
            float(yaw_des),
            float(yaw_rate_des),
        )

    def _apply_command_z_leash(self, p_des, v_des):
        if self.command_z_leash is None or p_des[2] <= self.p[2]:
            return p_des, v_des

        z_limited = min(p_des[2], self.p[2] + self.command_z_leash)
        if z_limited >= p_des[2]:
            return p_des, v_des

        p_limited = p_des.copy()
        v_limited = v_des.copy()
        p_limited[2] = z_limited
        v_limited[2] = min(0.0, v_limited[2])
        return p_limited, v_limited

    def _attitude_control(self, R, R_des, e_R, yaw_rate_des, dt):
        if self.attitude_control_mode == "ardupilot_pid":
            return self._ardupilot_pid_control(R, R_des, e_R, yaw_rate_des, dt)

        if self.attitude_control_mode == "cascade_pid":
            eR_d = (e_R - self.prev_eR) / dt
            self.prev_eR = e_R
            w_des = -1.0 * (self.Kr @ e_R + self.Kir @ self.eR_int + self.Kdr @ eR_d)
            w_des[2] += yaw_rate_des
            self.last_w_des = w_des.copy()

            e_w = self.w - w_des
            self.ew_int += e_w * dt
            ew_d = (e_w - self.prev_ew) / dt
            self.prev_ew = e_w

            tau_cmd = -1.0 * (self.Kw @ e_w + self.Kiw @ self.ew_int + self.Kdw @ ew_d)
            return tau_cmd, e_w

        w_des = np.array([0.0, 0.0, yaw_rate_des])
        w_des_body = R.T @ R_des @ w_des
        self.last_w_des = w_des_body.copy()
        e_w = self.w - w_des_body

        if self.attitude_control_mode == "pd":
            return -(self.Kr @ e_R) - (self.Kw @ e_w), e_w
        if self.attitude_control_mode != "lqr":
            carb.log_warn("Unknown attitude_control_mode, falling back to lqr: " + str(self.attitude_control_mode))

        error_rotation = R_des.T @ R
        omega_hat = self.hat(self.w)
        eI_hat = self.hat(self.eR_int)
        e_omega_matrix = self.lqr_kRP * error_rotation + self.lqr_kRI * eI_hat - error_rotation @ omega_hat
        e_omega = self.vee(self.skew(e_omega_matrix))

        x = np.concatenate((self.eR_int, e_R, e_omega))
        u_M = -(self.lqr_gain @ x)
        return self._lqr_virtual_input_to_moment(error_rotation, R, R_des, omega_hat, u_M), e_w

    def _ardupilot_pid_control(self, R, R_des, e_R, yaw_rate_des, dt):
        # ArduPilot's custom PID backend uses attitude error -> body-rate target,
        # then rate PID output directly as normalized motor roll/pitch/yaw input.
        attitude_error = -e_R
        target_rate = self.ap_angle_p * attitude_error
        target_rate[2] += yaw_rate_des
        self.last_w_des = target_rate.copy()

        rate_error = target_rate - self.w
        filtered_error = self._first_order_filter(
            self.ap_filtered_rate_error,
            rate_error,
            self.ap_rate_filter_hz,
            dt,
        )
        filtered_derivative = (filtered_error - self.ap_prev_filtered_rate_error) / dt
        self.ap_prev_filtered_rate_error = filtered_error.copy()
        self.ap_filtered_rate_error = filtered_error.copy()
        self.ap_prev_rate_error = rate_error.copy()

        self.ap_rate_int += self.ap_rate_i * filtered_error * dt
        self.ap_rate_int = np.clip(self.ap_rate_int, -self.ap_rate_imax, self.ap_rate_imax)

        motor_out = self.ap_rate_p * filtered_error
        motor_out += self.ap_rate_int
        motor_out += self.ap_rate_d * filtered_derivative
        return motor_out, -rate_error

    def _apm_xy_accel(self, pos_des_xy, vel_des_xy, accel_des_xy, dt):
        vel_target_xy = self.apm_xy_pos_p * (pos_des_xy - self.p[:2]) + vel_des_xy
        vel_error = vel_target_xy - self.v[:2]
        filtered_error = self._first_order_filter(
            self.apm_xy_filtered_vel_error,
            vel_error,
            self.apm_xy_vel_filter_hz,
            dt,
        )
        vel_error_d = (filtered_error - self.apm_xy_prev_filtered_vel_error) / dt
        self.apm_xy_prev_filtered_vel_error = filtered_error.copy()
        self.apm_xy_filtered_vel_error = filtered_error.copy()

        self.apm_xy_vel_int += self.apm_xy_vel_i * filtered_error * dt
        self.apm_xy_vel_int = np.clip(self.apm_xy_vel_int, -self.apm_xy_vel_imax, self.apm_xy_vel_imax)

        accel_xy = self.apm_xy_vel_p * filtered_error
        accel_xy += self.apm_xy_vel_int
        accel_xy += self.apm_xy_vel_d * vel_error_d
        return accel_xy + accel_des_xy

    def _apm_z_throttle(self, pos_des_z, vel_des_z, accel_des_z, dt, takeoff_window=False):
        vel_target_z = self.apm_z_pos_p * (pos_des_z - self.p[2]) + vel_des_z
        accel_target_z = self.apm_z_vel_p * (vel_target_z - self.v[2]) + accel_des_z
        accel_target_z = float(np.clip(accel_target_z, -self.apm_z_accel_limit, self.apm_z_accel_limit))

        if not self.apm_z_use_accel_feedback:
            throttle = self.hover_percentage + self.hover_percentage * accel_target_z / self.g
            return float(np.clip(throttle, 0.0, 1.0))

        accel_error = accel_target_z - self.a[2]
        filtered_error = self._first_order_filter(
            self.apm_z_filtered_accel_error,
            accel_error,
            self.apm_z_accel_filter_hz,
            dt,
        )
        accel_error_d = (filtered_error - self.apm_z_prev_filtered_accel_error) / dt
        self.apm_z_prev_filtered_accel_error = float(filtered_error)
        self.apm_z_filtered_accel_error = float(filtered_error)

        p_out = self.apm_z_accel_p * filtered_error
        d_out = self.apm_z_accel_d * accel_error_d
        throttle_without_i = self.hover_percentage + p_out + d_out
        throttle_with_i = throttle_without_i + self.apm_z_accel_int
        throttle_saturated = throttle_with_i <= 0.0 or throttle_with_i >= 1.0
        drives_away_from_limit = (
            (throttle_with_i >= 1.0 and filtered_error > 0.0)
            or (throttle_with_i <= 0.0 and filtered_error < 0.0)
        )
        if takeoff_window and filtered_error < 0.0:
            self.apm_z_accel_int = max(0.0, self.apm_z_accel_int)
        elif not throttle_saturated or not drives_away_from_limit:
            self.apm_z_accel_int += self.apm_z_accel_i * filtered_error * dt
            self.apm_z_accel_int = float(
                np.clip(self.apm_z_accel_int, -self.apm_z_accel_imax, self.apm_z_accel_imax)
            )

        return float(np.clip(throttle_without_i + self.apm_z_accel_int, 0.0, 1.0))

    def _in_takeoff_window(self, pos_des_z):
        return (pos_des_z - self.p[2]) > self.takeoff_error_gate and self.p[2] < self.takeoff_altitude_gate

    def _takeoff_xy_scale(self, pos_des_z):
        if pos_des_z <= self.p[2] or self.p[2] >= self.takeoff_altitude_gate:
            return 1.0
        return float(np.clip(self.p[2] / self.takeoff_altitude_gate, 0.25, 1.0))

    def _angle_boost_value(self, value, R, R_des):
        ez = np.array([0.0, 0.0, 1.0])
        cos_tilt_current = float(np.dot(R[:, 2], ez))
        inverted_factor = float(np.clip(10.0 * cos_tilt_current, 0.0, 1.0))
        cos_tilt_target = float(np.dot(R_des[:, 2], ez))
        boost_factor = 1.0 / float(np.clip(cos_tilt_target, self.angle_boost_min_cos, 1.0))
        return value * inverted_factor * boost_factor

    def _filter_throttle(self, throttle, dt):
        filtered = self._first_order_filter(
            self.filtered_throttle,
            float(throttle),
            self.throttle_filter_hz,
            dt,
        )
        self.filtered_throttle = float(filtered)
        return float(np.clip(self.filtered_throttle, 0.0, 1.0))

    def _build_lqr_gain(self):
        if self.lqr_q.shape != (9,):
            raise ValueError("lqr_q must contain 9 values for [eI, eR, eOmega]")
        if self.lqr_ru.shape != (3,):
            raise ValueError("lqr_ru must contain 3 input weights")

        identity = np.eye(3)
        zeros = np.zeros((3, 3))
        kRP = self.lqr_kRP
        kRI = self.lqr_kRI

        # This follows the 9-state construction in base_control.pdf 2.3.2:
        # x = [integral attitude error, attitude error, residual/rate error].
        # The paper then solves a continuous Riccati equation for u_M.
        a_matrix = np.block(
            [
                [zeros, identity, zeros],
                [kRI * identity, kRP * identity, -identity],
                [kRP * kRI * identity, (kRP * kRP + kRI) * identity, -kRP * identity],
            ]
        )
        b_matrix = np.vstack((zeros, zeros, -identity))
        q_matrix = np.diag(self.lqr_q)
        ru_matrix = np.diag(self.lqr_ru)

        p_matrix = solve_continuous_are(a_matrix, b_matrix, q_matrix, ru_matrix)
        return np.linalg.solve(ru_matrix, b_matrix.T @ p_matrix)

    def _lqr_virtual_input_to_moment(self, error_rotation, R, R_des, omega_hat, u_M):
        inertia_omega = self.inertia @ self.w
        gyro_term = np.cross(inertia_omega, self.w)
        angular_accel_drift = np.linalg.solve(self.inertia, gyro_term)

        m_cmd_1 = error_rotation @ omega_hat @ omega_hat
        m_cmd_1 += error_rotation @ self.hat(angular_accel_drift)

        mapped = R.T @ R_des @ (-m_cmd_1 + self.hat(u_M))
        return self.inertia @ self.vee(self.skew(mapped))

    def _limit_tilt(self, a_cmd):
        norm = np.linalg.norm(a_cmd)
        if norm < 1e-6:
            return np.array([0.0, 0.0, 1.0])

        b3 = a_cmd / norm
        ez = np.array([0.0, 0.0, 1.0])
        cos_tilt = float(np.dot(b3, ez))
        cos_max = float(np.cos(self.max_tilt))
        if cos_tilt >= cos_max:
            return b3

        lateral = b3 - cos_tilt * ez
        lateral_norm = np.linalg.norm(lateral)
        if lateral_norm < 1e-6:
            return ez
        return cos_max * ez + np.sin(self.max_tilt) * lateral / lateral_norm

    @staticmethod
    def _first_order_filter(previous, current, cutoff_hz, dt):
        cutoff_hz = np.asarray(cutoff_hz, dtype=float)
        alpha = (2.0 * np.pi * cutoff_hz * dt) / (1.0 + 2.0 * np.pi * cutoff_hz * dt)
        filtered = previous + alpha * (current - previous)
        return np.where(cutoff_hz <= 0.0, current, filtered)

    @staticmethod
    def _desired_rotation(b3_des, yaw_des):
        b1c = np.array([np.cos(yaw_des), np.sin(yaw_des), 0.0])
        b2 = np.cross(b3_des, b1c)
        if np.linalg.norm(b2) < 1e-6:
            b1c = np.array([-np.sin(yaw_des), np.cos(yaw_des), 0.0])
            b2 = np.cross(b3_des, b1c)
        b2 = b2 / np.linalg.norm(b2)
        b1 = np.cross(b2, b3_des)
        return np.column_stack((b1, b2, b3_des))

    @staticmethod
    def vee(S):
        return np.array([-S[1, 2], S[0, 2], -S[0, 1]])

    @staticmethod
    def hat(v):
        return np.array(
            [
                [0.0, -v[2], v[1]],
                [v[2], 0.0, -v[0]],
                [-v[1], v[0], 0.0],
            ]
        )

    @staticmethod
    def skew(A):
        return 0.5 * (A - A.T)

    def reset_statistics(self):
        self.total_time = 0.0
        self.time_vector = []
        self.position_over_time = []
        self.desired_position_over_time = []
        self.position_error_over_time = []
        self.velocity_error_over_time = []
        self.attitude_error_over_time = []
        self.attitude_rate_error_over_time = []
        self.attitude_over_time = []
        self.desired_attitude_over_time = []
        self.angular_rate_over_time = []
        self.desired_angular_rate_over_time = []
        self.control_over_time = []

    @staticmethod
    def _matrix_to_euler_xyz(matrix):
        return Rotation.from_matrix(matrix).as_euler("XYZ", degrees=False)
