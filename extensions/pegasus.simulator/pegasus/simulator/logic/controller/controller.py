# Imports to be able to log to the terminal with fancy colors
import carb

# Imports from the Pegasus library
from pegasus.simulator.logic.state import State

# Auxiliary scipy and numpy modules
import numpy as np
from scipy.spatial.transform import Rotation
import os


class Controller:

    def __init__(self, results_file: str = None):

        # The current state of the vehicle expressed in the inertial frame (in ENU)
        self.p = np.zeros((3,))  # The vehicle position
        self.R: Rotation = Rotation.identity()  # The vehicle attitude
        self.w = np.zeros((3,))  # The angular velocity of the vehicle
        self.v = np.zeros((3,))  # The linear velocity of the vehicle in the inertial frame
        self.a = np.zeros((3,))  # The linear acceleration of the vehicle in the inertial frame

        # Define the dynamic parameters for the vehicle
        self.m = 1.30  # Mass in Kg
        self.g = 9.81  # The gravity acceleration ms^-2
        self.hover_percentage = 0.48
        self.acc2thr = self.g / self.hover_percentage

        # Auxiliar variable, so that we only start sending motor commands once we get the state of the vehicle
        self.reveived_first_state = False

        # Lists used for analysing performance statistics
        if results_file is not None:
            self.results_files = os.path.join(os.path.dirname(os.path.abspath(__file__)), results_file)
        else:
            self.results_files = None

        self.total_time = 0.0

        # Reset the lists used for analysing performance statistics
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

        # Check if we should save the statistics to some file or not
        if self.results_files is None:
            return

        statistics = dict()
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
        self.a = state.linear_acceleration

        self.reveived_first_state = True

    def update(self, dt: float, cmd: list):
        """Method that implements the control. This method will be called by the simulation on every physics step

        Args:
            dt (float): The time elapsed between the previous and current function calls (s).
        """

        pass

    @staticmethod
    def vee(S):
        """Auxiliary function that computes the 'v' map which takes elements from so(3) to R^3.

        Args:
            S (np.array): A matrix in so(3)
        """
        return np.array([-S[1, 2], S[0, 2], -S[0, 1]])

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
