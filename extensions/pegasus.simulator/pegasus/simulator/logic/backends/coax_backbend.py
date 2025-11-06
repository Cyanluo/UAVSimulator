"""
| File: ros1_backend.py
| Author: LRX (cyanluorx@gmail.com)
| Description: File that implements the ROS1 Backend for communication/control with/of the vehicle simulation through ROS1 topics
| License: BSD-3-Clause. Copyright (c) 2025, LRX. All rights reserved.
"""

# Make sure the ROS1 extension is enabled
import carb
from isaacsim.core.utils import extensions
extensions.disable_extension("isaacsim.ros2.bridge")
extensions.enable_extension("isaacsim.ros1.bridge")

# ROS1 imports
import rospy
from std_msgs.msg import Float64
from geometry_msgs.msg import TransformStamped
from sensor_msgs.msg import Imu, MagneticField, NavSatFix, NavSatStatus
from geometry_msgs.msg import PoseStamped, TwistStamped, AccelStamped
from rosgraph_msgs.msg import Clock
from nav_msgs.msg import Odometry
from pegasus.simulator.logic.interface.pegasus_interface import PegasusInterface, MultirotorState

# TF imports
# Check if these libraries exist in the system
try:
    import tf2_ros
    tf2_ros_loaded = True
except ImportError:
    carb.log_warn("TF2 ROS not installed. Will not publish TFs with the ROS1 backend")
    tf2_ros_loaded = False

import isaacsim.core.utils.prims as prims_utils
import isaacsim.core.utils.transformations as transformations_utils
from scipy.spatial.transform import Rotation as R

from pegasus.simulator.logic.backends.backend import Backend
from pegasus.simulator.logic.controller.position_controller import trajController

# Import the replicatore core module used for writing graphical data to ROS 2
import omni
import omni.graph.core as og
import omni.replicator.core as rep
import omni.syntheticdata._syntheticdata as sd

import rosgraph

import numpy as np

from quadrotor_msgs.msg import PositionCommand


class ROS1Backend(Backend):

    def __init__(self, sim_app, vehicle_id: int, num_rotors=4, config: dict = {}):
        """Initialize the ROS1 Camera class

        Args:
            camera_prim_path (str): Path to the camera prim. Global path when it starts with `/`, else local to vehicle prim path
            config (dict): A Dictionary that contains all the parameters for configuring the ROS1Camera - it can be empty or only have some of the parameters used by the ROS1Camera.

        Examples:
            The dictionary default parameters are

            >>> {"namespace": "drone"                           # Namespace to append to the topics
            >>>  "pose_topic": "state/pose",                    # Position and attitude of the vehicle in ENU
            >>>  "twist_topic": "state/twist",                  # Linear and angular velocities in the body frame of the vehicle
            >>>  "twist_inertial_topic": "state/twist_inertial" # Linear velocity of the vehicle in the inertial frame
            >>>  "accel_topic": "state/accel",                  # Linear acceleration of the vehicle in the inertial frame
            >>>  "odom_topic": "state/odom",                    # Odometry of the vehicle from base_link frame to inertial frame
            >>>  "imu_topic": "sensors/imu",                    # IMU data
            >>>  "mag_topic": "sensors/mag",                    # Magnetometer data
            >>>  "gps_topic": "sensors/gps",                    # GPS data
            >>>  "gps_vel_topic": "sensors/gps_twist",          # GPS velocity data
            >>>  "pub_graphical_sensors": True,                 # Publish the graphical sensors
            >>>  "pub_sensors": True,                           # Publish the sensors
            >>>  "pub_state": True,                             # Publish the state of the vehicle
            >>>  "pub_tf": False,                               # Publish the TF of the vehicle
            >>>  "pub_clock": True,                             # Publish the clock topics 
            >>>  "sub_control": False,                           # Subscribe to the control topics
            >>>  "pos_cmd_topic": "cmd/position",               # Position command
            >>>  "result_file": None,                           # Result of controller for debug
        """
        super().__init__(config=config)

        self.sim_app = sim_app
        if not rosgraph.is_master_online():
            carb.log_error("Please run roscore before executing this script")
            sim_app.close()
            exit()

        self.pg = PegasusInterface()
        # Save the configurations for this backend
        self._id = vehicle_id
        self._num_rotors = num_rotors
        self._namespace = config.get("namespace", "drone" + str(vehicle_id))

        # Save what whould be published/subscribed
        self._pub_graphical_sensors = config.get("pub_graphical_sensors", True)
        self._pub_sensors = config.get("pub_sensors", True)
        self._pub_state = config.get("pub_state", True)
        self._sub_control = config.get("sub_control", False)
        self._pub_clock = config.get("pub_clock", False)

        # Check if the tf2_ros library is loaded and if the flag is set to True
        self._pub_tf = config.get("pub_tf", True)

        # Start the actual ROS1 setup here
        try:
            rospy.init_node("simulator_vehicle_" + str(vehicle_id),  disable_signals=True)
        except:
            # If rospy is already initialized, just ignore the exception
            pass

        # Initialize the publishers and subscribers
        self.initialize_publishers(config)
        self.initialize_subscribers(config)

        # Create a dictionary that will store the writers for the graphical sensors
        # NOTE: this is done this way, because the writers move data from the GPU->CPU and then publish it to ROS1
        # in a separate thread. This is done to avoid blocking the simulation
        self.graphical_sensors_writers = {}
        
        # Setup zero input reference for the thrusters
        self.input_ref = [0.0 for i in range(self._num_rotors)]

        # -----------------------------------------------------
        # Initialize the static and dynamic tf broadcasters
        # -----------------------------------------------------
        if self._pub_tf:

            # Initiliaze the static tf broadcaster for the sensors
            self.tf_static_broadcaster = tf2_ros.StaticTransformBroadcaster()

            # Initialize the dynamic tf broadcaster for the position of the body of the vehicle (base_link) with respect to the inertial frame (map - ENU) expressed in the inertil frame (map - ENU)
            self.tf_broadcaster = tf2_ros.TransformBroadcaster()
        
        self.state = None

        if self._sub_control:
            self.cmd = None
            self.controller = trajController(results_file=config.get("result_file", None))

    
    def initialize_publishers(self, config: dict):
        # ----------------------------------------------------- 
        # Create publishers for the state of the vehicle in ENU
        # -----------------------------------------------------
        if self._pub_state:
            self.pose_pub = rospy.Publisher(self._namespace + str(self._id) + "/" + config.get("pose_topic", "state/pose"), PoseStamped, queue_size=50)
            self.twist_pub = rospy.Publisher(self._namespace + str(self._id) + "/" + config.get("twist_topic", "state/twist"), TwistStamped, queue_size=50)
            self.twist_inertial_pub = rospy.Publisher(self._namespace + str(self._id) + "/" + config.get("twist_inertial_topic", "state/twist_inertial"), TwistStamped, queue_size=50)
            self.accel_pub = rospy.Publisher(self._namespace + str(self._id) + "/" + config.get("accel_topic", "state/accel"), AccelStamped, queue_size=50)
            self.odom_pub = rospy.Publisher(self._namespace + str(self._id) + "/" + config.get("odom_topic", "state/odom"), Odometry, queue_size=50)
        
        # -----------------------------------------------------
        # Create publishers for the sensors of the vehicle
        # -----------------------------------------------------
        if self._pub_sensors:
            self.imu_pub = rospy.Publisher(self._namespace + str(self._id) + "/" + config.get("imu_topic", "sensors/imu"), Imu, queue_size=50)
            self.mag_pub = rospy.Publisher(self._namespace + str(self._id) + "/" + config.get("mag_topic", "sensors/mag"), MagneticField, queue_size=50)
            self.gps_pub = rospy.Publisher(self._namespace + str(self._id) + "/" + config.get("gps_topic", "sensors/gps"), NavSatFix, queue_size=50)
            self.gps_vel_pub = rospy.Publisher(self._namespace + str(self._id) + "/" + config.get("gps_vel_topic", "sensors/gps_twist"), TwistStamped, queue_size=50)

        if self._pub_clock:
            clock_topic = "clock"
            self.clock_pub = rospy.Publisher(clock_topic, Clock, queue_size=1000)
            rospy.set_param("/use_sim_time", True)
        

    def initialize_subscribers(self, config: dict):
        if self._sub_control:
            self.pos_cmd_sub = rospy.Subscriber(self._namespace + str(self._id) + config.get("pos_cmd_topic", "/cmd/position"), PositionCommand, self.receive_position_cmd, queue_size=20)


    def receive_position_cmd(self, cmd: PositionCommand):
        l_cmd = [cmd.position,
                 cmd.velocity,
                 cmd.acceleration,
                 cmd.jerk]
        
        self.cmd = [np.asarray([c.x, c.y, c.z]) for c in l_cmd] + [cmd.yaw, cmd.yaw_dot]

    def take_off(self, height):
        if self.vehicle.vehicle_state == MultirotorState.LAND and self.state is not None:
            self.cmd = [np.asarray([self.state.position[0], self.state.position[1], height]),
                        np.asarray([0, 0, 0.5]), np.asarray([0, 0, 0]),
                        np.asarray([0, 0, 0]), R.from_quat(self.state.attitude).as_euler("ZYX")[0], 0.1]
            return True
        return False

    def hold(self):
        self.cmd = [np.asarray([self.state.position[0], self.state.position[1], self.state.position[2]]),
                    np.asarray([0, 0, 0]), np.asarray([0, 0, 0]),
                    np.asarray([0, 0, 0]), R.from_quat(self.state.attitude).as_euler("ZYX")[0], 0]

    def send_static_transforms(self):
        # Create the transformation from base_link FLU (ROS standard) to base_link FRD (standard in airborn and marine vehicles)
        t = TransformStamped()
        t.header.stamp = rospy.Time.now()
        t.header.frame_id = self._namespace + '_' + 'base_link'
        t.child_frame_id = self._namespace + '_' + 'base_link_frd'

        # Converts from FLU to FRD
        t.transform.translation.x = 0.0
        t.transform.translation.y = 0.0
        t.transform.translation.z = 0.0
        t.transform.rotation.x = 1.0
        t.transform.rotation.y = 0.0
        t.transform.rotation.z = 0.0
        t.transform.rotation.w = 0.0

        self.tf_static_broadcaster.sendTransform(t)

        # Create the transform from map, i.e inertial frame (ROS standard) to map_ned (standard in airborn or marine vehicles)
        t = TransformStamped()
        t.header.stamp = rospy.Time.now()
        t.header.frame_id = "map"
        t.child_frame_id = "map_ned"
        
        # Converts ENU to NED
        t.transform.translation.x = 0.0
        t.transform.translation.y = 0.0
        t.transform.translation.z = 0.0
        t.transform.rotation.x = -0.7071068
        t.transform.rotation.y = -0.7071068
        t.transform.rotation.z = 0.0
        t.transform.rotation.w = 0.0

        self.tf_static_broadcaster.sendTransform(t)

        if self.vehicle != None:
            body_prim = prims_utils.get_prim_at_path(self.vehicle._stage_prefix + "/body")
            rotors_prim_path = prims_utils.find_matching_prim_paths(self.vehicle._stage_prefix + "/rotor*")
            
            graphical_sensors_prim_path = list()
            for e in self.vehicle._graphical_sensors:
                graphical_sensors_prim_path.append(e._stage_prim_path)
            
            for e in (rotors_prim_path + graphical_sensors_prim_path):
                trans_matrix = transformations_utils.get_relative_transform(prims_utils.get_prim_at_path(e), body_prim)
                trans, rot_q = transformations_utils.pose_from_tf_matrix(trans_matrix)

                t = TransformStamped()
                t.header.stamp = rospy.Time.now()
                t.header.frame_id = self._namespace + '_' + "base_link"
                t.child_frame_id = e.rpartition("/")[-1]
                t.transform.translation.x = trans[0]
                t.transform.translation.y = trans[1]
                t.transform.translation.z = trans[2]
                t.transform.rotation.x = rot_q[1]
                t.transform.rotation.y = rot_q[2]
                t.transform.rotation.z = rot_q[3]
                t.transform.rotation.w = rot_q[0]

                self.tf_static_broadcaster.sendTransform(t)

                if t.child_frame_id.startswith('camera'):
                    rot_q = R.from_quat([rot_q[1], rot_q[2], rot_q[3], rot_q[0]])
                    rot_q *= R.from_euler('xyz', [180, 0, 0], degrees=True)
                    rot_q = rot_q.as_quat()
                    t.child_frame_id += '_ros'
                    t.transform.rotation.x = rot_q[0]
                    t.transform.rotation.y = rot_q[1]
                    t.transform.rotation.z = rot_q[2]
                    t.transform.rotation.w = rot_q[3]

                    self.tf_static_broadcaster.sendTransform(t)

    def update_state(self, state):
        """
        Method that when implemented, should handle the receivel of the state of the vehicle using this callback
        """
        self.state = state

        if self._pub_clock:
            self.clock_pub.publish(rospy.Time.from_sec(self.pg.world.current_time))

        if self._sub_control:
            self.controller.update_state(state)

        # Publish the state of the vehicle only if the flag is set to True
        if not self._pub_state:
            return

        pose = PoseStamped()
        twist = TwistStamped()
        twist_inertial = TwistStamped()
        accel = AccelStamped()
        odom = Odometry()

        # Update the header
        pose.header.stamp = rospy.Time.now()
        twist.header.stamp = pose.header.stamp
        twist_inertial.header.stamp = pose.header.stamp
        accel.header.stamp = pose.header.stamp
        odom.header.stamp = pose.header.stamp

        pose.header.frame_id = "map"
        twist.header.frame_id = self._namespace + "_" + "base_link"
        twist_inertial.header.frame_id = "map"
        accel.header.frame_id = "map"
        odom.header.frame_id = "map"
        odom.child_frame_id = self._namespace + "_" + "base_link"

        # Fill the position and attitude of the vehicle in ENU
        pose.pose.position.x = state.position[0]
        pose.pose.position.y = state.position[1]
        pose.pose.position.z = state.position[2]

        pose.pose.orientation.x = state.attitude[0]
        pose.pose.orientation.y = state.attitude[1]
        pose.pose.orientation.z = state.attitude[2]
        pose.pose.orientation.w = state.attitude[3]

        # Fill the linear and angular velocities in the body frame of the vehicle
        twist.twist.linear.x = state.linear_body_velocity[0]
        twist.twist.linear.y = state.linear_body_velocity[1]
        twist.twist.linear.z = state.linear_body_velocity[2]

        twist.twist.angular.x = state.angular_velocity[0]
        twist.twist.angular.y = state.angular_velocity[1]
        twist.twist.angular.z = state.angular_velocity[2]

        # Fill the linear velocity of the vehicle in the inertial frame
        twist_inertial.twist.linear.x = state.linear_velocity[0]
        twist_inertial.twist.linear.y = state.linear_velocity[1]
        twist_inertial.twist.linear.z = state.linear_velocity[2]

        # Fill the linear acceleration in the inertial frame
        accel.accel.linear.x = state.linear_acceleration[0]
        accel.accel.linear.y = state.linear_acceleration[1]
        accel.accel.linear.z = state.linear_acceleration[2]

        odom.pose.pose = pose.pose
        odom.twist.twist = twist_inertial.twist

        # Publish the messages containing the state of the vehicle
        self.pose_pub.publish(pose)
        self.twist_pub.publish(twist)
        self.twist_inertial_pub.publish(twist_inertial)
        self.accel_pub.publish(accel)
        self.odom_pub.publish(odom)

        # Update the dynamic tf broadcaster with the current position of the vehicle in the inertial frame
        if self._pub_tf:
            t = TransformStamped()
            t.header.stamp = pose.header.stamp
            t.header.frame_id = "map"
            t.child_frame_id = self._namespace + '_' + 'base_link'
            t.transform.translation.x = state.position[0]
            t.transform.translation.y = state.position[1]
            t.transform.translation.z = state.position[2]
            t.transform.rotation.x = state.attitude[0]
            t.transform.rotation.y = state.attitude[1]
            t.transform.rotation.z = state.attitude[2]
            t.transform.rotation.w = state.attitude[3]
            self.tf_broadcaster.sendTransform(t)

            self.send_static_transforms()

    def rotor_callback(self, ros_msg: Float64, rotor_id):
        # Update the reference for the rotor of the vehicle
        self.input_ref[rotor_id] = float(ros_msg.data)

    def update_sensor(self, sensor_type: str, data):
        """
        Method that when implemented, should handle the receival of sensor data
        """

        # Only process sensor data if the flag is set to True
        if not self._pub_sensors:
            return

        if sensor_type == "IMU":
            self.update_imu_data(data)
        elif sensor_type == "GPS":
            self.update_gps_data(data)
        elif sensor_type == "Magnetometer":
            self.update_mag_data(data)
        else:
            pass

    def update_graphical_sensor(self, sensor_type: str, data):
        """
        Method that when implemented, should handle the receival of graphical sensor data
        """

        # Only process graphical sensor data if the flag is set to True
        if not self._pub_graphical_sensors:
            return
        if sensor_type == "MonocularCamera":
            self.update_monocular_camera_data(data)
        elif sensor_type == "Lidar":
            self.update_lidar_data(data)
        else:
            pass

    def update_imu_data(self, data):

        msg = Imu()

        # Update the header
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = self._namespace + '_' + "base_link_frd"
        
        # Update the angular velocity (NED + FRD)
        msg.angular_velocity.x = data["angular_velocity"][0]
        msg.angular_velocity.y = data["angular_velocity"][1]
        msg.angular_velocity.z = data["angular_velocity"][2]
        
        # Update the linear acceleration (NED)
        msg.linear_acceleration.x = data["linear_acceleration"][0]
        msg.linear_acceleration.y = data["linear_acceleration"][1]
        msg.linear_acceleration.z = data["linear_acceleration"][2]

        # Publish the message with the current imu state
        self.imu_pub.publish(msg)

    def update_gps_data(self, data):

        msg = NavSatFix()
        msg_vel = TwistStamped()

        # Update the headers
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = "map_ned"
        msg_vel.header.stamp = msg.header.stamp
        msg_vel.header.frame_id = msg.header.frame_id

        # Update the status of the GPS
        status_msg = NavSatStatus()
        status_msg.status = 0 # unaugmented fix position
        status_msg.service = 1 # GPS service
        msg.status = status_msg

        # Update the latitude, longitude and altitude
        msg.latitude = data["latitude"]
        msg.longitude = data["longitude"]
        msg.altitude = data["altitude"]

        # Update the velocity of the vehicle measured by the GPS in the inertial frame (NED)
        msg_vel.twist.linear.x = data["velocity_north"]
        msg_vel.twist.linear.y = data["velocity_east"]
        msg_vel.twist.linear.z = data["velocity_down"]

        # Publish the message with the current GPS state
        self.gps_pub.publish(msg)
        self.gps_vel_pub.publish(msg_vel)

    def update_mag_data(self, data):
        
        msg = MagneticField()

        # Update the headers
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = "base_link_frd"

        msg.magnetic_field.x = data["magnetic_field"][0]
        msg.magnetic_field.y = data["magnetic_field"][1]
        msg.magnetic_field.z = data["magnetic_field"][2]

        # Publish the message with the current magnetic data
        self.mag_pub.publish(msg)

    def update_monocular_camera_data(self, data):

        # Check if the camera name exists in the writers dictionary
        if data["camera_name"] not in self.graphical_sensors_writers:
            self.add_monocular_camera_writter(data)


    def add_monocular_camera_writter(self, data):

        # List all the available writers: print(rep.writers.WriterRegistry._writers)
        render_prod_path = data["camera"]._render_product_path

        # Create the writer for the rgb camera
        rv = omni.syntheticdata.SyntheticData.convert_sensor_type_to_rendervar(sd.SensorType.Rgb.name)
        writer = rep.writers.get(rv + "ROS1PublishImage")
        
        writer.initialize(nodeNamespace=self._namespace + str(self._id), topicName=data["camera_name"] + "/color/image_raw", frameId=data["camera_name"], queueSize=1)
        writer.attach([render_prod_path])

        # Add the writer to the dictionary
        self.graphical_sensors_writers[data["camera_name"]] = [writer]

        # Check if depth is enabled, if so, set the depth properties
        if "depth" in data:

            # Create the writer for the depth camera
            rv = omni.syntheticdata.SyntheticData.convert_sensor_type_to_rendervar(
                sd.SensorType.DistanceToImagePlane.name
            )
            writer_depth = rep.writers.get(rv + "ROS1PublishImage")
            
            writer_depth.initialize(nodeNamespace=self._namespace + str(self._id), topicName=data["camera_name"] + "/depth", frameId=data["camera_name"], queueSize=1)
            writer_depth.attach([render_prod_path])

            # Add the writer to the dictionary
            self.graphical_sensors_writers[data["camera_name"]].append(writer_depth)

        # Create a writer for publishing the camera info
        stereo_offset = [0.0, 0.0]
        writer_info = rep.writers.get("ROS1PublishCameraInfo")

        writer_info.initialize(
            frameId=data["camera_name"],
            nodeNamespace=self._namespace + str(self._id),
            queueSize=1,
            topicName=data["camera_name"] + "/color/camera_info",
            stereoOffset=stereo_offset,
        )
        writer_info.attach([render_prod_path])

        # Add the writer to the dictionary
        self.graphical_sensors_writers[data["camera_name"]].append(writer_info)

        gate_path = omni.syntheticdata.SyntheticData._get_node_path("PostProcessDispatch" + "IsaacSimulationGate", render_prod_path)

        # Set step input of the Isaac Simulation Gate nodes upstream of ROS publishers to control their execution rate
        og.Controller.attribute(gate_path + ".inputs:step").set(int(60/data["frequency"]))

    def update_lidar_data(self, data):

        # Check if the lidar name exists in the writers dictionary
        if data["lidar_name"] not in self.graphical_sensors_writers:
            self.add_lidar_writter(data)
    
    def add_lidar_writter(self, data):

        # List all the available writers: print(rep.writers.WriterRegistry._writers)
        render_prod_path = rep.create.render_product(data["stage_prim_path"], [1, 1], name=data["lidar_name"])

        # Create the writer for the lidar
        writer = rep.writers.get("RtxLidar" + "ROS1PublishPointCloud")
        writer.initialize(nodeNamespace=self._namespace + str(self._id), topicName=data["lidar_name"] + "/pointcloud", frameId=data["lidar_name"])
        writer.attach([render_prod_path])

        # Add the writer to the dictionary
        self.graphical_sensors_writers[data["lidar_name"]] = [writer]

        # Create the writer for publishing a laser scan message along with the point cloud
        writer = rep.writers.get("RtxLidar" + "ROS1PublishLaserScan")
        writer.initialize(nodeNamespace=self._namespace + str(self._id), topicName=data["lidar_name"] + "/laserscan", frameId=data["lidar_name"])
        writer.attach([render_prod_path])
        self.graphical_sensors_writers[data["lidar_name"]].append(writer)

    def input_reference(self):
        """
        Method that is used to return the latest target angular velocities to be applied to the vehicle

        Returns:
            A list with the target angular velocities for each individual rotor of the vehicle
        """
        # return self.input_ref
        return self.input_ref

    def update(self, dt: float):
        """
        Method that when implemented, should be used to update the state of the backend and the information being sent/received
        from the communication interface. This method will be called by the simulation on every physics step
        """

        if self._sub_control:
            f, torques = self.controller.update(dt, self.cmd)
            if self.vehicle:
                self.input_ref = self.vehicle.force_and_torques_to_velocities(f, torques)


    def start(self):
        """
        Method that when implemented should handle the begining of the simulation of vehicle
        """
        # Reset the reference for the thrusters
        self.input_ref = [0.0 for i in range(self._num_rotors)]
        
        if self._sub_control:
            self.controller.start()

    def stop(self):
        """
        Method that when implemented should handle the stopping of the simulation of vehicle
        """
        # Reset the reference for the thrusters
        self.cmd = None
        self.input_ref = [0.0 for i in range(self._num_rotors)]

        if self._sub_control:
            self.controller.stop()

    def reset(self):
        """
        Method that when implemented, should handle the reset of the vehicle simulation to its original state
        """
        # Reset the reference for the thrusters
        self.cmd = None
        self.input_ref = [0.0 for i in range(self._num_rotors)]

        if self._sub_control:
            self.controller.reset()