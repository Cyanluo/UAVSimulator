#!/usr/bin/env python
"""
| File: 8_camera_vehicle.py
| License: BSD-3-Clause. Copyright (c) 2024, Marcelo Jacinto. All rights reserved.
| Description: This files serves as an example on how to build an app that makes use of the Pegasus API, 
| where the data is send/received through mavlink, the vehicle is controled using mavlink and
| camera data is sent to ROS1 topics at the same time.
"""

# Imports to start Isaac Sim from this script
import carb
from isaacsim import SimulationApp

# Start Isaac Sim's simulation environment
# Note: this simulation app must be instantiated right after the SimulationApp import, otherwise the simulator will crash
# as this is the object that will load all the extensions and load the actual simulator.
simulation_app = SimulationApp({"headless": False})

# -----------------------------------
# The actual script should start here
# -----------------------------------
import omni
import omni.timeline
from omni.isaac.core.world import World
import omni.usd
from pxr import UsdLux, Gf

# Import the Pegasus API for simulating drones
from pegasus.simulator.params import ROBOTS, SIMULATION_ENVIRONMENTS
from pegasus.simulator.logic.graphical_sensors.monocular_camera import MonocularCamera
from pegasus.simulator.logic.backends.ros1_backend import ROS1Backend
from pegasus.simulator.logic.vehicles.multirotor import Multirotor, MultirotorConfig
from pegasus.simulator.logic.interface.pegasus_interface import PegasusInterface

# Auxiliary scipy and numpy modules
from scipy.spatial.transform import Rotation

class PegasusApp:
    """
    A Template class that serves as an example on how to build a simple Isaac Sim standalone App.
    """

    def __init__(self):
        """
        Method that initializes the PegasusApp and is used to setup the simulation environment.
        """

        # Acquire the timeline that will be used to start/stop the simulation
        self.timeline = omni.timeline.get_timeline_interface()

        # Start the Pegasus Interface
        self.pg = PegasusInterface()

        # Acquire the World, .i.e, the singleton that controls that is a one stop shop for setting up physics,
        # spawning asset primitives, etc.
        self.pg._world = World(**self.pg._world_settings)
        self.world = self.pg.world

        # Launch one of the worlds provided by NVIDIA
        self.pg.load_environment(SIMULATION_ENVIRONMENTS["Default Environment"])

        # DomeLight
        stage = omni.usd.get_context().get_stage()
        light_path = "/World/DomeLight"
        domeLight = UsdLux.DomeLight.Define(stage, light_path)
        domeLight.CreateIntensityAttr(1000)
        domeLight.CreateColorAttr(Gf.Vec3f(1.0, 1.0, 1.0))

        from omni.isaac.core.objects import DynamicCuboid
        import numpy as np
        cube_2 = self.world.scene.add(
            DynamicCuboid(
                prim_path="/new_cube_2",
                name="cube_1",
                position=np.array([-3.0, 0, 2.0]),
                scale=np.array([1.0, 1.0, 1.5]),
                size=1.0,
                color=np.array([255, 0, 0]),
            )
        )

        cube_3 = self.world.scene.add(
            DynamicCuboid(
                prim_path="/new_cube_3",
                name="cube_2",
                position=np.array([-1.9, 1.3, 2.0]),
                scale=np.array([1.0, 1.0, 1.0]),
                size=1.0,
                color=np.array([255, 0, 0]),
            )
        )

        # Create the vehicle
        # Try to spawn the selected robot in the world to the specified namespace
        config_multirotor = MultirotorConfig()
        config_multirotor.backends = [
            ROS1Backend(vehicle_id=1, 
                        sim_app= simulation_app,
                        config={
                            "namespace": 'drone', 
                            "pub_sensors": True,
                            "pub_graphical_sensors": True,
                            "pub_state": True,
                            "sub_control": True,
                            "pub_tf": False,
                            "pub_clock": True})]

        # Create a camera and lidar sensors
        config_multirotor.graphical_sensors = [MonocularCamera("camera", config={"update_rate": 60.0, "depth": True,})] # Lidar("lidar")
        
        Multirotor(
            "/World/quadrotor",
            ROBOTS['Iris'],
            0,
            [0.0, 0.0, 0.07],
            Rotation.from_euler("XYZ", [0.0, 0.0, 0.0], degrees=True).as_quat(),
            config=config_multirotor,
            collision_check=True
        )
    
        # Reset the simulation environment so that all articulations (aka robots) are initialized
        self.world.reset()

        # Auxiliar variable for the timeline callback example
        self.stop_sim = False

    def run(self):
        """
        Method that implements the application main loop, where the physics steps are executed.
        """

        # Start the simulation
        self.timeline.play()

        # The "infinite" loop
        while simulation_app.is_running() and not self.stop_sim:
            # Update the UI of the app and perform the physics step
            self.world.step(render=True)

        # Cleanup and stop
        carb.log_warn("PegasusApp Simulation App is closing.")
        self.timeline.stop()
        simulation_app.close()

def main():

    # Instantiate the template app
    pg_app = PegasusApp()

    # Run the application loop
    pg_app.run()

if __name__ == "__main__":
    main()
