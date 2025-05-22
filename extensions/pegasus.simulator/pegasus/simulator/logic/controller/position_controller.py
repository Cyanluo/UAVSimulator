# Imports to be able to log to the terminal with fancy colors
import carb

# Imports from the Pegasus library
from pegasus.simulator.logic.state import State

# Auxiliary scipy and numpy modules
import numpy as np
from scipy.spatial.transform import Rotation

class trajController:
    """A nonlinear controller class. This controlers is well described in the papers
    
    [1] J. Pinto, B. J. Guerreiro and R. Cunha, "Planning Parcel Relay Manoeuvres for Quadrotors," 
    2021 International Conference on Unmanned Aircraft Systems (ICUAS), Athens, Greece, 2021, 
    pp. 137-145, doi: 10.1109/ICUAS51884.2021.9476757.
    [2] D. Mellinger and V. Kumar, "Minimum snap trajectory generation and control for quadrotors," 
    2011 IEEE International Conference on Robotics and Automation, Shanghai, China, 2011, 
    pp. 2520-2525, doi: 10.1109/ICRA.2011.5980409.
    """

    def __init__(self, 
        results_file: str=None, 
        Kp=[10.0, 10.0, 10.0],
        Kd=[8.5, 8.5, 8.5],
        Ki=[1.5, 1.5, 1.5],
        Kr=[3.5, 3.5, 3.5],
        Kw=[0.5, 0.5, 0.5]):

        # The current rotor references [rad/s]
        self.input_ref = [0.0, 0.0, 0.0, 0.0]

        # The current state of the vehicle expressed in the inertial frame (in ENU)
        self.p = np.zeros((3,))                   # The vehicle position
        self.R: Rotation = Rotation.identity()    # The vehicle attitude
        self.w = np.zeros((3,))                   # The angular velocity of the vehicle
        self.v = np.zeros((3,))                   # The linear velocity of the vehicle in the inertial frame
        self.a = np.zeros((3,))                   # The linear acceleration of the vehicle in the inertial frame

        # Define the control gains matrix for the outer-loop
        self.Kp = np.diag(Kp)
        self.Kd = np.diag(Kd)
        self.Ki = np.diag(Ki)
        self.Kr = np.diag(Kr)
        self.Kw = np.diag(Kw)

        self.int = np.array([0.0, 0.0, 0.0])

        # Define the dynamic parameters for the vehicle
        self.m = 1.50        # Mass in Kg
        self.g = 9.81       # The gravity acceleration ms^-2

        # Auxiliar variable, so that we only start sending motor commands once we get the state of the vehicle
        self.reveived_first_state = False

        self.total_time = 0

        # Lists used for analysing performance statistics
        self.results_files = results_file
        self.time_vector = []
        self.desired_position_over_time = []
        self.position_over_time = []
        self.position_error_over_time = []
        self.velocity_error_over_time = []
        self.atittude_error_over_time = []
        self.attitude_rate_error_over_time = []

    def reset(self):
        self.stop()
        self.start()

    def start(self):
        """
        Reset the control and trajectory index
        """
        self.reset_statistics()
        

    def stop(self):
        """
        Stopping the controller. Saving the statistics data for plotting later
        """

        self.reveived_first_state = False
        self.int = 0

        # Check if we should save the statistics to some file or not
        if self.results_files is None:
            return
        
        statistics = {}
        statistics["time"] = np.array(self.time_vector)
        statistics["p"] = np.vstack(self.position_over_time)
        statistics["desired_p"] = np.vstack(self.desired_position_over_time)
        statistics["ep"] = np.vstack(self.position_error_over_time)
        statistics["ev"] = np.vstack(self.velocity_error_over_time)
        statistics["er"] = np.vstack(self.atittude_error_over_time)
        statistics["ew"] = np.vstack(self.attitude_rate_error_over_time)
        np.savez(self.results_files, **statistics)
        carb.log_warn("Statistics saved to: " + self.results_files)

        self.reset_statistics()

    def update_state(self, state: State):
        """
        Method that updates the current state of the vehicle. This is a callback that is called at every physics step

        Args:
            state (State): The current state of the vehicle.
        """
        self.p = state.position
        self.R = Rotation.from_quat(state.attitude)
        self.w = state.angular_velocity
        self.v = state.linear_velocity

        self.reveived_first_state = True

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
        self.int = self.int +  (ep * dt)
        ei = self.int

        # Compute F_des term
        F_des = -(self.Kp @ ep) - (self.Kd @ ev) - (self.Ki @ ei) + np.array([0.0, 0.0, self.m * self.g]) + (self.m * a_ref)

        # Get the current axis Z_B (given by the last column of the rotation matrix)
        Z_B = self.R.as_matrix()[:,2]

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
        R_des = np.c_[X_b_des, Y_b_des, Z_b_des]
        R = self.R.as_matrix()

        # Compute the rotation error
        e_R = 0.5 * self.vee((R_des.T @ R) - (R.T @ R_des))

        # Compute an approximation of the current vehicle acceleration in the inertial frame (since we cannot measure it directly)
        self.a = (u_1 * Z_B) / self.m - np.array([0.0, 0.0, self.g])

        # Compute the desired angular velocity by projecting the angular velocity in the Xb-Yb plane
        # projection of angular velocity on xB − yB plane
        # see eqn (7) from [2].
        hw = (self.m / u_1) * (j_ref - np.dot(Z_b_des, j_ref) * Z_b_des) 
        
        # desired angular velocity
        w_des = np.array([-np.dot(hw, Y_b_des), 
                           np.dot(hw, X_b_des), 
                           yaw_rate_ref * Z_b_des[2]])

        # Compute the angular velocity error
        e_w = self.w - w_des

        # Compute the torques to apply on the rigid body
        tau = -(self.Kr @ e_R) - (self.Kw @ e_w)

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

        # Return desired force and torques
        return u_1, tau

    @staticmethod
    def vee(S):
        """Auxiliary function that computes the 'v' map which takes elements from so(3) to R^3.

        Args:
            S (np.array): A matrix in so(3)
        """
        return np.array([-S[1,2], S[0,2], -S[0,1]])
    
    def reset_statistics(self):
        self.total_time = 0.0

        # Reset the lists used for analysing performance statistics
        self.time_vector = []
        self.desired_position_over_time = []
        self.position_over_time = []
        self.position_error_over_time = []
        self.velocity_error_over_time = []
        self.atittude_error_over_time = []
        self.attitude_rate_error_over_time = []
