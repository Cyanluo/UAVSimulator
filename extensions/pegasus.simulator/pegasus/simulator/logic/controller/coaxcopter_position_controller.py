from pegasus.simulator.logic.controller.controller import Controller

# Auxiliary scipy and numpy modules
import numpy as np
from scipy.spatial.transform import Rotation

class trajController(Controller):

    def __init__(self,
         results_file: str = None,
         Kp=[10.0, 10.0, 7.0],
         Kd=[1.5, 1.5, 1.5],
         Ki=[0.2, 0.2, 0.2],
         Kr=[53.0, 75.0, 40.0],
         kir=[10.0, 10.0, 15.0],
         kdr=[0.1, 0.6, 0.05],
         Kw=[0.013, 0.013, 0.01],
         kiw=[0.002, 0.002, 0.0015],
         kdw=[0.000008, 0.000006, 0.00015]):

        super().__init__(results_file)

        # Define the control gains matrix for the outer-loop
        self.Kp = np.diag(Kp)
        self.Kd = np.diag(Kd)
        self.Ki = np.diag(Ki)
        self.Kr = np.diag(Kr)
        self.Kir = np.diag(kir)
        self.Kdr = np.diag(kdr)
        self.Kw = np.diag(Kw)
        self.Kiw = np.diag(kiw)
        self.Kdw = np.diag(kdw)

        self.int = np.array([0.0, 0.0, 0.0])
        self.eR_int = np.array([0.0, 0.0, 0.0])
        self.ew_int = np.array([0.0, 0.0, 0.0])

        self.pre_eR = np.array([0.0, 0.0, 0.0])
        self.pre_ew = np.array([0.0, 0.0, 0.0])

    def stop(self):
        """
        Stopping the controller. Saving the statistics data for plotting later
        """

        self.int = np.array([0.0, 0.0, 0.0])
        self.eR_int = np.array([0.0, 0.0, 0.0])
        self.ew_int = np.array([0.0, 0.0, 0.0])

        self.pre_eR = np.array([0.0, 0.0, 0.0])
        self.pre_ew = np.array([0.0, 0.0, 0.0])

        super().stop()

    def update(self, dt: float, cmd: list):
        """Method that implements the nonlinear control law and updates the target angular velocities for each rotor.
        This method will be called by the simulation on every physics step

        Args:
            dt (float): The time elapsed between the previous and current function calls (s).
        """

        if self.reveived_first_state == False or cmd == None:
            return 0, [0, 0, 0]

        # -------------------------------------------------
        # Update the references for the controller to track
        # -------------------------------------------------
        self.total_time += dt

        p_ref, v_ref, a_ref, j_ref, yaw_ref, yaw_rate_ref = cmd

        # -------------------------------------------------
        # Start the controller implementation
        # ------------------------------------------------

        # Compute the tracking errors
        ep = self.p - p_ref
        ev = self.v - v_ref
        self.int = self.int + (ep * dt)
        ei = self.int

        # Compute F_des term
        F_des = -(self.Kp @ ep) - (self.Kd @ ev) - (self.Ki @ ei) + np.array([0.0, 0.0, self.m * self.g]) + (
                    self.m * a_ref)

        # Get the current axis Z_B (given by the last column of the rotation matrix)
        Z_B = self.R.as_matrix()[:, 2]

        # Get the desired total thrust in Z_B direction (u_1)
        u_1 = F_des @ Z_B

        # Compute the desired body-frame axis Z_b
        Z_b_des = F_des / np.linalg.norm(F_des)

        # Compute X_C_des
        X_c_des = np.array([np.cos(yaw_ref), np.sin(yaw_ref), 0.0])

        # Compute Y_b_des
        Z_b_cross_X_c = np.cross(Z_b_des, X_c_des)
        Y_b_des = Z_b_cross_X_c / np.linalg.norm(Z_b_cross_X_c)

        # Compute X_b_des
        X_b_des = np.cross(Y_b_des, Z_b_des)

        # Compute the desired rotation R_des = [X_b_des | Y_b_des | Z_b_des]
        R_des = Rotation.from_euler("XYZ", [10.0, -10.0, 0.0], degrees=True).as_matrix()
        # R_des = np.c_[X_b_des, Y_b_des, Z_b_des]
        R = self.R.as_matrix()

        # Compute the rotation error
        e_R = 0.5 * self.vee((R_des.T @ R) - (R.T @ R_des))

        self.eR_int = self.eR_int + e_R * dt
        eR_d = (e_R - self.pre_eR) / dt
        self.pre_eR = e_R

        w_des = -1 * (self.Kr @ e_R + self.Kir @ self.eR_int + self.Kdr @ eR_d)
        # Compute the angular velocity error
        e_w = self.w - w_des
        self.ew_int = self.ew_int + e_w * dt
        ew_d = (e_w - self.pre_ew) / dt
        self.pre_ew = e_w

        # Compute the torques to apply on the rigid body
        tau = -1 * (self.Kw @ e_w + self.Kiw @ self.ew_int + self.Kdw @ ew_d)
        # ----------------------------
        # Statistics to save for later
        # ----------------------------
        self.time_vector.append(self.total_time)
        self.position_over_time.append(self.p)
        self.desired_position_over_time.append(p_ref)
        self.position_error_over_time.append(ep)
        self.velocity_error_over_time.append(ev)
        self.atittude_error_over_time.append(e_R)
        self.attitude_rate_error_over_time.append(e_w)

        u_1 = u_1 / self.acc2thr

        # Return desired force and torques
        # print("u_1, tau:", u_1, tau)
        np.clip(u_1, 0, 1)
        np.clip(tau, -1, 1)

        return u_1, tau