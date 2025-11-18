from pegasus.simulator.logic.backends.ros1_backend import ROS1Backend
import rospy
from geometry_msgs.msg import TransformStamped
from pegasus.simulator.logic.controller.coaxcopter_position_controller import trajController

class ROS1CoaxCopterBackend(ROS1Backend):
    def __init__(self, sim_app, vehicle_id: int, num_rotors=2, num_servo=2, config: dict = {}):
        self._num_servo = num_servo
        super().__init__(sim_app, vehicle_id, num_rotors, config, controller = trajController(config.get("result_file", None)))

    def init_input_reference(self):
        self.input_ref = dict()
        self.input_ref["rotor"] = [0.0 for i in range(self._num_rotors)]
        self.input_ref["servo"] = [0.0 for i in range(self._num_servo)]

    def send_static_transforms(self):
        pass
