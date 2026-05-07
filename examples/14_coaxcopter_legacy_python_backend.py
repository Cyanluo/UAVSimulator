#!/usr/bin/env python

import carb
from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})

import omni
import omni.timeline
from omni.isaac.core.world import World
import omni.usd
from pxr import UsdLux, Gf

from pegasus.simulator.params import ROBOTS, SIMULATION_ENVIRONMENTS
from pegasus.simulator.logic.interface.pegasus_interface import PegasusInterface
from pegasus.simulator.logic.vehicles.coaxcopter import CoaxCopter, CoaxCopterConfig
from pegasus.simulator.logic.graphical_sensors.monocular_camera import MonocularCamera
from pegasus.simulator.logic.backends.legacy_coax_python_backend import LegacyCoaxPythonBackend

from scipy.spatial.transform import Rotation


class PegasusApp:
    def __init__(self):
        self.timeline = omni.timeline.get_timeline_interface()
        self.pg = PegasusInterface()
        self.pg._world = World(**self.pg._world_settings)
        self.world = self.pg.world

        self.pg.load_environment(SIMULATION_ENVIRONMENTS["Curved Gridroom"])

        stage = omni.usd.get_context().get_stage()
        dome_light = UsdLux.DomeLight.Define(stage, "/World/DomeLight")
        dome_light.CreateIntensityAttr(1000)
        dome_light.CreateColorAttr(Gf.Vec3f(1.0, 1.0, 1.0))

        config_coaxcopter = CoaxCopterConfig()
        config_coaxcopter.backends = [
            LegacyCoaxPythonBackend(
                target_height=3.0,
                results_file="debug/legacy_python_backend_hover.npz",
            )
        ]
        config_coaxcopter.graphical_sensors = [
            MonocularCamera("camera", config={"update_rate": 60.0, "depth": True})
        ]

        CoaxCopter(
            "/World/coaxcopter",
            ROBOTS["dumbbel"],
            0,
            [3.3, 0.0, 1.0],
            Rotation.from_euler("XYZ", [0.0, 0.0, 0.0], degrees=True).as_quat(),
            config=config_coaxcopter,
            collision_check=True,
        )

        self.pg.set_viewport_camera([5.0, 9.0, 6.5], [3.3, 0.0, 2.5])
        self.world.reset()
        self.stop_sim = False

    def run(self):
        self.timeline.play()
        while simulation_app.is_running() and not self.stop_sim:
            self.world.step(render=True)

        carb.log_warn("PegasusApp Simulation App is closing.")
        self.timeline.stop()
        simulation_app.close()


def main():
    PegasusApp().run()


if __name__ == "__main__":
    main()
