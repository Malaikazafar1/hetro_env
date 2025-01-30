"""Script demonstrating the joint use of simulation and control.

The simulation is run by a `CtrlAviary` environment.
The control is given by the PID implementation in `DSLPIDControl`.

Example
-------
In a terminal, run as:

    $ python pid.py

Notes
-----
The drones move, at different altitudes, along cicular trajectories 
in the X-Y plane, around point (0, -.3).

"""
import os
import time
import argparse
from datetime import datetime
import pdb
import math
import random
import numpy as np
import pybullet as p
import matplotlib.pyplot as plt

from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary 
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.control.SimplePIDControl import SimplePIDControl
from gym_pybullet_drones.control.CTBRControl import CTBRControl
from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.utils.utils import sync, str2bool

DEFAULT_DRONES = DroneModel("racer")
DEFAULT_NUM_DRONES = 1
DEFAULT_PHYSICS = Physics("pyb")
DEFAULT_GUI = True
DEFAULT_RECORD_VISION = False
DEFAULT_PLOT = True
DEFAULT_USER_DEBUG_GUI = False
DEFAULT_OBSTACLES = True
DEFAULT_SIMULATION_FREQ_HZ = 240
DEFAULT_CONTROL_FREQ_HZ = 48
DEFAULT_DURATION_SEC = 150
DEFAULT_OUTPUT_FOLDER = 'results'
DEFAULT_COLAB = False
DEFAULT_NUM_CARS = 3 # Default number of cars

def run(
        drone=DEFAULT_DRONES,
        num_drones=DEFAULT_NUM_DRONES,
        NUM_CARS = DEFAULT_NUM_CARS, ##############################cars
        physics=DEFAULT_PHYSICS,
        gui=DEFAULT_GUI,
        record_video=DEFAULT_RECORD_VISION,
        plot=DEFAULT_PLOT,
        user_debug_gui=DEFAULT_USER_DEBUG_GUI,
        obstacles=DEFAULT_OBSTACLES,
        simulation_freq_hz=DEFAULT_SIMULATION_FREQ_HZ,
        control_freq_hz=DEFAULT_CONTROL_FREQ_HZ,
        duration_sec=DEFAULT_DURATION_SEC,
        output_folder=DEFAULT_OUTPUT_FOLDER,
        colab=DEFAULT_COLAB
        ):
    
    # Initial positions, targets, and orientations for the cars ##############################cars
    INIT_XYZS_C = np.array([[i*2, i*2, 0] for i in range(NUM_CARS)])  # Example initial positions
    INIT_RPYS_C = np.array([[0, 0, 0] for _ in range(NUM_CARS)])  # Example initial orientations

    NUM_WP_CARS = 1000 #control_freq_hz*PERIOD  # Number of waypoints for cars along the trajectory
    TARGET_POS_CARS = np.zeros((NUM_WP_CARS, 3, NUM_CARS))  # 3D waypoints for each car

    # Generate trajectory for each car (moving along x from 0 to 1)
    for i in range(NUM_WP_CARS):
        for j in range(NUM_CARS):
            # Move the car along the x-axis from 0 to 1, while keeping y and z constant
            TARGET_POS_CARS[i, 0, j] = i / (NUM_WP_CARS - 1)  # Gradually move from 0 to 1 on x-axis
            TARGET_POS_CARS[i, 1, j] = i / (NUM_WP_CARS - 1)  # Constant y-coordinate (can be adjusted)
            TARGET_POS_CARS[i, 2, j] = 0  # Constant z-coordinate (can be adjusted)

    # Initialize the car waypoint counters
    car_wp_counters = np.zeros(NUM_CARS, dtype=int)

    #### Initialize the simulation #############################
    H = 0.2
    H_STEP = 0.1
    R = .3
    INIT_XYZS = np.array([[R*np.cos((i/6)*2*np.pi+np.pi/2), R*np.sin((i/6)*2*np.pi+np.pi/2)-R, H+i*H_STEP] for i in range(num_drones)])
    INIT_RPYS = np.array([[0, 0,  i * (np.pi/2)/num_drones] for i in range(num_drones)])

    #### Initialize a circular trajectory ######################
    PERIOD = 10
    NUM_WP = control_freq_hz*PERIOD
    TARGET_POS = np.zeros((NUM_WP,3))
    for i in range(NUM_WP):
        TARGET_POS[i, :] = R*np.cos((i/NUM_WP)*(2*np.pi)+np.pi/2)+INIT_XYZS[0, 0], R*np.sin((i/NUM_WP)*(2*np.pi)+np.pi/2)-R+INIT_XYZS[0, 1], 0 # INIT_XYZS[0, 2] + i*2
    wp_counters = np.array([int((i*NUM_WP/6)%NUM_WP) for i in range(num_drones)])

    ### Debug trajectory ######################################
    ### Uncomment alt. target_pos in .computeControlFromState()
    INIT_XYZS = np.array([[.3 * i, 0, .5] for i in range(num_drones)])
    INIT_RPYS = np.array([[0, 0,  i * (np.pi/3)/num_drones] for i in range(num_drones)])
    NUM_WP = control_freq_hz*15
    TARGET_POS = np.zeros((NUM_WP,3))
    scale = 2
    for i in range(NUM_WP):
        if i < NUM_WP/6:
            TARGET_POS[i, :] = ((i*6)/NUM_WP)*scale, 0, 0.5*(i*6)/NUM_WP 
        elif i < 2 * NUM_WP/6:
            TARGET_POS[i, :] = (1 - ((i-NUM_WP/6)*6)/NUM_WP)*scale, 0, 0.5 - 0.5*((i-NUM_WP/6)*6)/NUM_WP
        elif i < 3 * NUM_WP/6:
            TARGET_POS[i, :] = 0, (((i-2*NUM_WP/6)*6)/NUM_WP)*scale, 0.5*((i-2*NUM_WP/6)*6)/NUM_WP
        elif i < 4 * NUM_WP/6:
            TARGET_POS[i, :] = 0, (1 - ((i-3*NUM_WP/6)*6)/NUM_WP)*scale, 0.5 - 0.5*((i-3*NUM_WP/6)*6)/NUM_WP
        elif i < 5 * NUM_WP/6:
            TARGET_POS[i, :] = (((i-4*NUM_WP/6)*6)/NUM_WP)*scale, (((i-4*NUM_WP/6)*6)/NUM_WP)*scale, 0.5*((i-4*NUM_WP/6)*6)/NUM_WP
        elif i < 6 * NUM_WP/6:
            TARGET_POS[i, :] = (1 - ((i-5*NUM_WP/6)*6)/NUM_WP)*scale, (1 - ((i-5*NUM_WP/6)*6)/NUM_WP)*scale, 0.5 - 0.5*((i-5*NUM_WP/6)*6)/NUM_WP
    wp_counters = np.array([0 for i in range(num_drones)])

    #### Create the environment ################################
    env = CtrlAviary(drone_model=drone,
                        num_drones=num_drones,
                        initial_xyzs=INIT_XYZS,
                        initial_rpys=INIT_RPYS,
                        physics=physics,
                        neighbourhood_radius=10,
                        pyb_freq=simulation_freq_hz,
                        ctrl_freq=control_freq_hz,
                        gui=gui,
                        record=record_video,
                        obstacles=obstacles,
                        user_debug_gui=user_debug_gui,
                        NUM_CARS=NUM_CARS,  # Pass num_cars dynamically  
                        INIT_XYZS_C=INIT_XYZS_C,  # Using global car initial positions
                        INIT_RPYS_C=INIT_RPYS_C
                        )
    #### Obtain the PyBullet Client ID from the environment ####
    PYB_CLIENT = env.getPyBulletClient()

    #### Initialize the initial target position for each car#################################

    target_pos_car = []
    for j in range(NUM_CARS):
            target_pos_car1 = TARGET_POS_CARS[car_wp_counters[j], :, j]
            target_pos_car.append(target_pos_car1.tolist())  # Append the target position as a list
            env.target = np.array(target_pos_car)

    #### Initialize the logger #################################
    logger = Logger(logging_freq_hz=control_freq_hz,
                    num_drones=num_drones,
                    output_folder=output_folder,
                    colab=colab
                    )


    #### Initialize the controllers ############################
    if drone in [DroneModel.CF2X, DroneModel.CF2P]:
        ctrl = [DSLPIDControl(drone_model=drone) for i in range(num_drones)]
    elif drone in [DroneModel.RACE]:
        ctrl = [DSLPIDControl(drone_model=drone) for i in range(num_drones)]
    
    #### Run the simulation ####################################
    action = np.zeros((num_drones,4))
    
    START = time.time()
    for i in range(0, int(duration_sec*env.CTRL_FREQ)):

        #### Make it rain rubber ducks #############################
        # if i/env.SIM_FREQ>5 and i%10==0 and i/env.SIM_FREQ<10: p.loadURDF("duck_vhacd.urdf", [0+random.gauss(0, 0.3),-0.5+random.gauss(0, 0.3),3], p.getQuaternionFromEuler([random.randint(0,360),random.randint(0,360),random.randint(0,360)]), physicsClientId=PYB_CLIENT)

        #### Step the simulation ###################################

        obs, reward, terminated, truncated, info = env.step(action)
        
        #### Compute control for the current way point #############
        for j in range(num_drones):
            action[j, :], _, _ = ctrl[j].computeControlFromState(control_timestep=env.CTRL_TIMESTEP,
                                                                    state=obs[j],
                                                                    # target_pos=np.hstack([TARGET_POS[wp_counters[j], 0:2], INIT_XYZS[j, 2]]),
                                                                    target_pos=INIT_XYZS[j, :] + TARGET_POS[wp_counters[j], :],
                                                                    target_rpy=INIT_RPYS[j, :]
                                                                    )

        #### Go to the next way point and loop #####################
        for j in range(num_drones):
            wp_counters[j] = wp_counters[j] + 1 if wp_counters[j] < (NUM_WP-1) else 0
        
        #### Go to the next way point and loop  for each car #####################
        for counter in range(NUM_WP_CARS):
            current_target_pos_car = []
            # Loop through each car
            for j in range(NUM_CARS):
                target_pos_car1 = TARGET_POS_CARS[counter, :, j]
                current_target_pos_car.append(target_pos_car1.tolist())  # Append the target position as a list
            env.target = np.array(current_target_pos_car)
            car_wp_counters[j] = (car_wp_counters[j] + 1) % NUM_WP_CARS

        #### Log the simulation ####################################
        for j in range(num_drones):
            logger.log(drone=j,
                       timestamp=i/env.CTRL_FREQ,
                       state=obs[j],
                       control=np.hstack([TARGET_POS[wp_counters[j], 0:2], INIT_XYZS[j, 2], INIT_RPYS[j, :], np.zeros(6)])
                       # control=np.hstack([INIT_XYZS[j, :]+TARGET_POS[wp_counters[j], :], INIT_RPYS[j, :], np.zeros(6)])
                       )

        #### Printout ##############################################
        env.render()

        #### Sync the simulation ###################################
        if gui:
            sync(i, START, env.CTRL_TIMESTEP)

    #### Close the environment #################################
    env.close()

    #### Save the simulation results ###########################
    logger.save()
    logger.save_as_csv("pid") # Optional CSV save

    #### Plot the simulation results ###########################
    if plot:
        logger.plot()

if __name__ == "__main__":
    #### Define and parse (optional) arguments for the script ##
    parser = argparse.ArgumentParser(description='Helix flight script using CtrlAviary and DSLPIDControl')
    parser.add_argument('--drone',              default=DEFAULT_DRONES,                 type=DroneModel,    help='Drone model (default: CF2X)', metavar='', choices=DroneModel)
    parser.add_argument('--num_drones',         default=DEFAULT_NUM_DRONES,             type=int,           help='Number of drones (default: 3)', metavar='')
    parser.add_argument('--physics',            default=DEFAULT_PHYSICS,                type=Physics,       help='Physics updates (default: PYB)', metavar='', choices=Physics)
    parser.add_argument('--gui',                default=DEFAULT_GUI,                    type=str2bool,      help='Whether to use PyBullet GUI (default: True)', metavar='')
    parser.add_argument('--record_video',       default=DEFAULT_RECORD_VISION,          type=str2bool,      help='Whether to record a video (default: False)', metavar='')
    parser.add_argument('--plot',               default=DEFAULT_PLOT,                   type=str2bool,      help='Whether to plot the simulation results (default: True)', metavar='')
    parser.add_argument('--user_debug_gui',     default=DEFAULT_USER_DEBUG_GUI,         type=str2bool,      help='Whether to add debug lines and parameters to the GUI (default: False)', metavar='')
    parser.add_argument('--obstacles',          default=DEFAULT_OBSTACLES,              type=str2bool,      help='Whether to add obstacles to the environment (default: True)', metavar='')
    parser.add_argument('--simulation_freq_hz', default=DEFAULT_SIMULATION_FREQ_HZ,     type=int,           help='Simulation frequency in Hz (default: 240)', metavar='')
    parser.add_argument('--control_freq_hz',    default=DEFAULT_CONTROL_FREQ_HZ,        type=int,           help='Control frequency in Hz (default: 48)', metavar='')
    parser.add_argument('--duration_sec',       default=DEFAULT_DURATION_SEC,           type=int,           help='Duration of the simulation in seconds (default: 5)', metavar='')
    parser.add_argument('--output_folder',      default=DEFAULT_OUTPUT_FOLDER,          type=str,           help='Folder where to save logs (default: "results")', metavar='')
    parser.add_argument('--colab',              default=DEFAULT_COLAB,                  type=bool,          help='Whether example is being run by a notebook (default: "False")', metavar='')
    parser.add_argument('--NUM_CARS',           default=DEFAULT_NUM_CARS,               type=int,           help='Number of cars (default: 2)', metavar='') ##############################cars
    ARGS = parser.parse_args()

    run(**vars(ARGS))
