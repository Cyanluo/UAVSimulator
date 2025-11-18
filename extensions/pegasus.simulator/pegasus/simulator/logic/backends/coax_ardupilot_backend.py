from pegasus.simulator.logic.backends.ardupilot_mavlink_backend import ArduPilotMavlinkBackend, ArduPilotMavlinkBackendConfig
import numpy as np

class CoaxArduPilotBackend(ArduPilotMavlinkBackend):

    def __init__(self, config: ArduPilotMavlinkBackendConfig = ArduPilotMavlinkBackendConfig()):
        super().__init__(config)

    def input_reference(self):
        input_cmd = super().input_reference()

        ret = dict()
        ret["rotor"] = [input_cmd[4], input_cmd[5]]
        ret["servo"] = [-3.14*0.5*np.clip(-1.0 + input_cmd[i] * 2, -0.8, 0.8) for i in range(1, -1, -1)]
        ret["servo"][0] = -ret["servo"][0]

        return ret