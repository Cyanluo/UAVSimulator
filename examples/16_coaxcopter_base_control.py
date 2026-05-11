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
from pegasus.simulator.logic.backends.base_control_backend import BaseControlBackend
from pegasus.simulator.logic.controller.base_control_controller import BaseControlController
from pegasus.simulator.logic.controller.coax_ardupilot_mixer import CoaxArduPilotMixer
from pegasus.simulator.logic.controller.coax_lm_mixer import CoaxLMMixer

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
            BaseControlBackend(
                target_height=3.0,
                results_file="debug/base_control_hover.npz",
                trajectory_enabled=True,
                trajectory_delay=0.3,
                trajectory_radii=(1.8, 1.5, 0.78),
                trajectory_period=9.5,
                trajectory_ramp_time=1.6,
                trajectory_log_interval=1.0,
                controller=BaseControlController(
                    results_file="debug/base_control_hover.npz",
                    # Mission Planner values from the APM coaxial SITL tune.
                    # XY path uses PSC_POSXY_P followed by PSC_VELXY_PID.
                    Kp=(0.0, 0.0, 0.0),
                    Kv=(0.0, 0.0, 0.0),
                    Kvi=(0.0, 0.0, 0.0),
                    Kvd=(0.0, 0.0, 0.0),
                    apm_xy_pos_p=1.25,
                    apm_xy_vel_p=1.55,
                    apm_xy_vel_i=0.32,
                    apm_xy_vel_d=0.0,
                    apm_xy_vel_imax=1.2,
                    apm_xy_vel_filter_hz=10.0,
                    ap_angle_p=(4.6, 4.45, 2.6),
                    ap_rate_p=(0.026, 0.034, 0.045),
                    ap_rate_i=(0.035, 0.024, 0.055),
                    # The Isaac angular velocity signal is much noisier than APM's filtered gyro path.
                    ap_rate_d=(0.0, 0.0, 0.0),
                    ap_rate_imax=(0.12, 0.12, 0.12),
                    ap_rate_filter_hz=(18.0, 18.0, 1.6),
                    apm_z_pos_p=0.8,
                    apm_z_vel_p=3.2,
                    apm_z_accel_p=0.030,
                    apm_z_accel_i=0.120,
                    apm_z_accel_d=0.0,
                    apm_z_accel_imax=0.08,
                    apm_z_accel_filter_hz=20.0,
                    apm_z_accel_limit=1.4,
                    apm_z_use_accel_feedback=True,
                    command_z_leash=1.0,
                    throttle_filter_hz=4.0,
                    direct_lateral_force_ratio=0.6,
                    direct_lateral_force_ratio_y=0.6,
                    lateral_force_limit=0.07,
                    lateral_force_filter_hz=1.0,
                    hover_percentage=0.21,
                    max_tilt_deg=32.0,
                    yaw_rate_gain=0.0,
                    output_mode="normalized",
                ),
                mixer=CoaxLMMixer(
                    throttle_hover=0.21,
                    wrench_weights=(0.08, 0.08, 3.0, 1.0, 1.35, 2.0),
                ),
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
    pg_app = PegasusApp()
    pg_app.run()


if __name__ == "__main__":
    main()
