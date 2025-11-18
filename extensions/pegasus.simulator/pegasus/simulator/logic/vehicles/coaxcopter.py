import numpy as np

# The vehicle interface
from pegasus.simulator.logic.vehicles.vehicle import Vehicle

# Mavlink interface
from pegasus.simulator.logic.backends.px4_mavlink_backend import PX4MavlinkBackend, PX4MavlinkBackendConfig

# Sensors and dynamics setup
from pegasus.simulator.logic.dynamics import LinearDrag
from pegasus.simulator.logic.thrusters import QuadraticThrustCurve
from pegasus.simulator.logic.sensors import Barometer, IMU, Magnetometer, GPS
from pegasus.simulator.logic.interface.pegasus_interface import MultirotorState, PegasusInterface

class CoaxCopterConfig:
    """
    A data class that is used for configuring a CoaxCopter
    """

    def __init__(self):
        """
        Initialization of the MultirotorConfig class
        """

        # Stage prefix of the vehicle when spawning in the world
        self.stage_prefix = "coaxcopter"

        # The USD file that describes the visual aspect of the vehicle (and some properties such as mass and moments of inertia)
        self.usd_file = ""

        self.num_servo = 2

        # The default thrust curve for a quadrotor and dynamics relating to drag
        # 15 inch
        self.thrust_curve = QuadraticThrustCurve(config={
                                "num_rotors": 2,
                                "rotor_constant": [3.92e-5, 3.92e-5],
                                "rolling_moment_coefficient": [1.6e-6, 1.6e-6],
                                "rot_dir": [1, 1],
                                "force_dir": [1, -1],
                                "min_rotor_velocity": [0, 0],       # rad/s
                                "max_rotor_velocity": [900, 900], # rad/s
                                })

        self.drag = LinearDrag([0.50, 0.30, 0.0])

        # The default sensors for a quadrotor
        self.sensors = [Barometer(), IMU(), Magnetometer(), GPS()]

        # The default graphical sensors for a quadrotor
        self.graphical_sensors = []

        # The default omnigraphs for a quadrotor
        self.graphs = []

        # The backends for actually sending commands to the vehicle. By default use mavlink (with default mavlink configurations)
        # [Can be None as well, if we do not desired to use PX4 with this simulated vehicle]. It can also be a ROS2 backend
        # or your own custom Backend implementation!
        self.backends = [PX4MavlinkBackend(config=PX4MavlinkBackendConfig())]


class CoaxCopter(Vehicle):
    """CoaxCopter class - It defines a base interface for creating a coaxCopter
    """
    def __init__(
        self,
        # Simulation specific configurations
        stage_prefix: str = "coaxcopter",
        usd_file: str = "",
        vehicle_id: int = 0,
        # Spawning pose of the vehicle
        init_pos=[0.0, 0.0, 0.07],
        init_orientation=[0.0, 0.0, 0.0, 1.0],
        collision_check=False,
        config=CoaxCopterConfig(),
    ):
        """Initializes the coaxCopter object

        Args:
            stage_prefix (str): The name the vehicle will present in the simulator when spawned. Defaults to "quadrotor".
            usd_file (str): The USD file that describes the looks and shape of the vehicle. Defaults to "".
            vehicle_id (int): The id to be used for the vehicle. Defaults to 0.
            init_pos (list): The initial position of the vehicle in the inertial frame (in ENU convention). Defaults to [0.0, 0.0, 0.07].
            init_orientation (list): The initial orientation of the vehicle in quaternion [qx, qy, qz, qw]. Defaults to [0.0, 0.0, 0.0, 1.0].
            config (MultirotorConfig, optional): Defaults to MultirotorConfig().
        """

        # 1. Initiate the Vehicle object itself
        super().__init__(stage_prefix, usd_file, init_pos, init_orientation,
                         config.sensors, config.graphical_sensors,
                         config.graphs, config.backends, collision_check=collision_check,
                         base_name="/base_link")

        # 2. Setup the dynamics of the system - get the thrust curve of the vehicle from the configuration
        self._thrusters = config.thrust_curve
        self._drag = config.drag
        self._num_servo = config.num_servo
        self.pg = PegasusInterface()
        self.take_off_flag = False
        self.camera_pos = np.array([5.0, 4.0, 0.3])
        self.camera_target = np.array([3.3, 0.0, 0.3])
        self._target_take_off_height = 0

        # vehicle state: 0:land  1:flying  2:collision
        self._vehicle_state = MultirotorState.LAND

    @property
    def vehicle_state(self):
        return self._vehicle_state

    def stop(self):
        """In this case we do not need to do anything extra when the simulation stops"""
        self._vehicle_state = MultirotorState.LAND
        super().stop()

    def update(self, dt: float):
        """
        Method that computes and applies the forces to the vehicle in simulation based on the motor speed.
        This method must be implemented by a class that inherits this type. This callback
        is called on every physics step.

        Args:
            dt (float): The time elapsed between the previous and current function calls (s).
        """
        # if self.pg.world.current_time > 2:
        if self._vehicle_state == MultirotorState.LAND:
            self.take_off(3.0)

        if self._vehicle_state == MultirotorState.TAKE_OFF:
            if self._state.position[2] > (self._target_take_off_height-0.01):
                self._backends[0].hold()
                self._vehicle_state = MultirotorState.FLYING

        # Get the desired angular velocities for each rotor from the first backend (can be mavlink or other) expressed in rad/s
        desired_drive_val = dict()
        if len(self._backends) != 0:
            desired_drive_val = self._backends[0].input_reference()
        else:
            desired_drive_val["rotor"] = [0.0 for i in range(self._thrusters._num_rotors)]
            desired_drive_val["servo"] = [0.0 for i in range(self._num_servo)]

        pos = self._state.position
        v1 = self.camera_target - self.camera_pos
        v2 = pos - self.camera_pos
        # 计算夹角（单位：度）
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        cos_angle = np.clip(cos_angle, -1.0, 1.0)  # 防止数值误差
        angle = np.degrees(np.arccos(cos_angle))

        # 计算距离
        dist = np.linalg.norm(self._state.position - self.camera_pos)

        # 判断是否需要更新视角
        if angle > 20 or dist > 40:
            self.camera_pos = np.array([pos[0] + 8, pos[1] + 8, pos[2] + 2])
            self.camera_target = pos
            self.pg.set_viewport_camera(
                self.camera_pos.tolist(),
                pos.tolist()
            )

        # Input the desired rotor velocities in the thruster model
        self._thrusters.set_input_reference(desired_drive_val["rotor"])

        # Get the desired forces to apply to the vehicle
        forces_z, _, rolling_moment = self._thrusters.update(self._state, dt)

        rotors_name = ["/motor_up_link", "/motor_down_link"]

        # Apply force to each rotor
        for i, name in enumerate(rotors_name):

            # Apply the force in Z on the rotor frame
            self.apply_force([0.0, 0.0, forces_z[i]], body_part=name)

            # Apply the torque to the body frame of the vehicle that corresponds to the rolling moment
            self.apply_torque([0.0, 0.0, rolling_moment[i]], body_part=name)

            # Generate the rotating propeller visual effect
            self.handle_propeller_visual(i, desired_drive_val["rotor"][i])

        self.set_joints_position_targets(desired_drive_val["servo"], ["servo_up_joint", "servo_down_joint"])

        # Compute the total linear drag force to apply to the vehicle's body frame
        drag = self._drag.update(self._state, dt)
        self.apply_force(drag, body_part=self._base_name)

        # Call the update methods in all backends
        for backend in self._backends:
            backend.update(dt)

        is_contact, value = self.in_contact()
        if is_contact and (self._vehicle_state == MultirotorState.FLYING or self._vehicle_state == MultirotorState.COLLISION):
            self._vehicle_state = MultirotorState.COLLISION
            print(value)
            for contact in value['contacts']:
                print(f" Contact: {contact['body0']} <--> {contact['body1']}")
            print("\r\n")
            self.reset()

    def reset(self):
        self._vehicle_state = MultirotorState.LAND
        super().reset()

    def handle_propeller_visual(self, rotor_number, speed: float):
        """
        Auxiliar method used to set the joint velocity of each rotor (for animation purposes) based on the
        amount of force being applied on each joint

        Args:
            rotor_number (int): The number of the rotor to generate the rotation animation
            force (float): The force that is being applied on that rotor
            articulation (_type_): The articulation group the joints of the rotors belong to
        """

        rotors_joint = ["motor_up_joint", "motor_down_joint"]

        self.set_joints_velocity_targets([speed * self._thrusters.rot_dir[rotor_number]], [rotors_joint[rotor_number]])

    def adjust_pair_float(self, a, b, dmax):
        """
        Ensure that the absolute difference between the two values is <= dmax
        while preserving their relative order.
        If the difference is already within the range, no adjustment is made.
        """
        swapped = False
        if a < b:
            a, b = b, a
            swapped = True

        d = a - b
        if d <= dmax:
            return (b, a) if swapped else (a, b)

        delta = d - dmax
        a_new = a - delta / 2.0
        b_new = b + delta / 2.0

        return (b_new, a_new) if swapped else (a_new, b_new)

    def decay_scalar(self, a, b, delta=0.2, b0=0.6, k=0.5, gamma=1.6):
        """
        b=0.6 -> y = a
        b < 0.6 -> |y| > |a|
        b > 0.6 -> |y| < |a|
        |y - a| <= delta
        """
        b = max(0.0, min(1.0, b))
        diff = b - b0
        scale = 1.0 - k * (np.sign(diff) if diff != 0 else 0.0) * (abs(diff) ** gamma)
        y_cand = a * scale
        y = max(a - delta, min(a + delta, y_cand))
        return y

    def force_and_torques_to_velocities(self, force: float, torque: np.ndarray):

        if force == torque[0] == torque[1] == torque[2] == 0:
            desired_drive_val = dict()
            desired_drive_val["rotor"] = [0.0 for i in range(self._thrusters._num_rotors)]
            desired_drive_val["servo"] = [0.0 for i in range(self._num_servo)]
            return  desired_drive_val

        out_yaw_thrust = torque[2]
        up_thrust = np.clip(force-0.5*out_yaw_thrust, 0.2, 1)
        down_thrust = np.clip(force+0.5*out_yaw_thrust, 0.2, 1)

        up_thrust, down_thrust = self.adjust_pair_float(up_thrust, down_thrust, 0.4)
        print("up_thrust, down_thrust:", up_thrust, down_thrust)

        up_thrust_vel = up_thrust * self._thrusters.max_rotor_velocity[0]
        down_thrust_vel = down_thrust * self._thrusters.max_rotor_velocity[1]

        desired_drive_val = dict()
        desired_drive_val["rotor"] = [up_thrust_vel, down_thrust_vel]

        up_angel = np.clip(self.decay_scalar(-torque[1]/0.6*0.5*3.14, force), -0.5*3.14*0.9, 0.5*3.14*0.9)
        down_angel = np.clip(self.decay_scalar(-torque[0]/0.6*0.5*3.14, force), -0.5*3.14*0.9, 0.5*3.14*0.9)
        desired_drive_val["servo"] = [up_angel, down_angel]

        return desired_drive_val

    def take_off(self, height):
        if len(self._backends) != 0:
            if self._backends[0].take_off(height):
                self._target_take_off_height = height
                self._vehicle_state = MultirotorState.TAKE_OFF
