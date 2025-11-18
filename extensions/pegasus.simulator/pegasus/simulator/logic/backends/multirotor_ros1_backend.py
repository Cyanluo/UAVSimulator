from pegasus.simulator.logic.backends.ros1_backend import ROS1Backend
import rospy
from geometry_msgs.msg import TransformStamped
from pegasus.simulator.logic.controller.multirotor_position_controller import trajController

class ROS1MultiRotorBackend(ROS1Backend):
    def __init__(self, sim_app, vehicle_id: int, num_rotors=4, config: dict = {}):
        super().__init__(sim_app, vehicle_id, num_rotors, config, controller = trajController(config.get("result_file", None)))

    def init_input_reference(self):
        self.input_ref = [0.0 for i in range(self._num_rotors)]

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
