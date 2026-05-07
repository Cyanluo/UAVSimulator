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
                takeoff_climb_rate=0.2,
                takeoff_z_leash=0.3,
                takeoff_delay=3.0,
                takeoff_throttle_slew_time=2.0,
                takeoff_spool_release_throttle=0.24,
                takeoff_spool_min_altitude=0.12,
                trajectory_enabled=True,
                trajectory_delay=0.5,
                trajectory_radii=(1.6, 1.2, 0.55),
                trajectory_period=12.0,
                trajectory_ramp_time=2.0,
                controller=BaseControlController(
                    results_file="debug/base_control_hover.npz",
                    attitude_control_mode="ardupilot_pid",
                    # Mission Planner values from the APM coaxial SITL tune.
                    # XY path uses PSC_POSXY_P followed by PSC_VELXY_PID.
                    Kp=(0.0, 0.0, 0.0),
                    Kv=(0.0, 0.0, 0.0),
                    Kvi=(0.0, 0.0, 0.0),
                    Kvd=(0.0, 0.0, 0.0),
                    apm_xy_pos_p=1.0,
                    apm_xy_vel_p=1.2,
                    apm_xy_vel_i=0.45,
                    apm_xy_vel_d=0.0,
                    apm_xy_vel_imax=1.5,
                    apm_xy_vel_filter_hz=20.0,
                    ap_angle_p=(4.0, 3.396, 3.0),
                    ap_rate_p=(0.020, 0.020, 0.080),
                    ap_rate_i=(0.040, 0.010, 0.100),
                    # The Isaac angular velocity signal is much noisier than APM's filtered gyro path.
                    ap_rate_d=(0.0, 0.0, 0.0),
                    ap_rate_imax=(0.15, 0.15, 0.25),
                    ap_rate_filter_hz=(20.0, 20.0, 2.5),
                    apm_z_pos_p=1.0,
                    apm_z_vel_p=5.0,
                    apm_z_accel_p=0.060,
                    apm_z_accel_i=0.120,
                    apm_z_accel_d=0.0,
                    apm_z_accel_imax=0.08,
                    apm_z_accel_filter_hz=20.0,
                    apm_z_accel_limit=0.8,
                    apm_z_use_accel_feedback=False,
                    command_z_leash=0.6,
                    throttle_filter_hz=4.0,
                    takeoff_min_throttle=0.205,
                    takeoff_altitude_gate=0.6,
                    takeoff_error_gate=0.2,
                    z_priority_takeoff=True,
                    hover_percentage=0.21,
                    max_tilt_deg=18.0,
                    yaw_rate_gain=0.0,
                    output_mode="normalized",
                ),
                mixer=CoaxArduPilotMixer(throttle_hover=0.21),
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
