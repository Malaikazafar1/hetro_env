import gymnasium as gym
import pybullet as p
import torch
from stable_baselines3 import PPO
from gym_pybullet_drones.envs.DynamicAgentTest import DynamicAgentTest
from gym_pybullet_drones.utils.enums import ObservationType, ActionType, DroneModel
import time
import numpy as np
import csv
import subprocess

# Define the environment parameters (these should match the parameters used during training)
AGGR_PHY_STEPS = 5
EPISODE_LEN_SEC = 10
DEFAULT_OBS = ObservationType('kin')  # 'kin' or 'rgb'
DEFAULT_ACT = ActionType('vel')       # 'rpm' or 'pid' or 'vel' or 'one_d_rpm' or 'one_d_pid'
DEFAULT_DRONE_MODEL = DroneModel.RACE
OUTPUT_FOLDER = 'results'  # Folder where to save logs

env_kwargs = dict(aggregate_phy_steps=AGGR_PHY_STEPS, obs=DEFAULT_OBS,
                  act=DEFAULT_ACT, episode_len_sec=EPISODE_LEN_SEC,
                  output_folder=OUTPUT_FOLDER, drone_model=DEFAULT_DRONE_MODEL, gui=True, freq=240)

# Load the trained model
model_path = "./results/working/PPO-01.22.2025_04.31.07/best_model.zip"
# model_path = "./results/PPO-01.22.2025_17.01.08/best_model.zip"
model = PPO.load(model_path)

# Create the environment
env = DynamicAgentTest(**env_kwargs)  

env.GATE_1_PARAMS = [0.0, 0.0, 1.5, -0/57.3, 1.57]
env.GATE_2_PARAMS = [0.0, 2.0, 1.5, -0/57.3, 1.57] 
env.OBSTACLE_1_PARAMS = [0.0, -1.0, 0, 0.20]
env.OBSTACLE_2_PARAMS = [-3.0, 1.0, 0, 0.20]
env.OBSTACLE_SINGLE_STATUS = False

# Reset the environment
obs, info = env.reset()

file = open('results/test_env/observations_test.csv', mode='w', newline='')
writer = csv.writer(file)
# Write the header
writer.writerow(['time', 'x', 'y', 'z', 'vx', 'vy', 'vz'])

start_time = time.time()
done = False
while not done:
    # Predict the action using the trained model
    action, _states = model.predict(obs, deterministic=True)  
    
    # Step the environment
    obs, reward, terminated, truncated, info = env.step(action)
    print(obs)
    # print('obs',obs)
    # print('action',action)
    x, y, z = obs[0,0], obs[0,1], obs[0,2]
    vx, vy, vz = obs[0,6], obs[0,7], obs[0,8]  
    
    # Calculate the elapsed time
    current_time = time.time() - start_time
    # Write the observations to the file
    writer.writerow([current_time, x, y, z, vx, vy, vz])
    
    # Set the camera position and orientation
    camera_distance = 1.2  # Adjust the zoom (distance of the camera from the target)
    camera_yaw = 0.0      # Camera angle along the z-axis (left-right)
    camera_pitch = -30   # Camera angle along the y-axis (up-down)
    camera_target = [x, y, z]  # Center the camera on the drone
    
    p.resetDebugVisualizerCamera(
        cameraDistance=camera_distance,
        cameraYaw=camera_yaw,
        cameraPitch=camera_pitch,
        cameraTargetPosition=camera_target
    )
    
    # Render the environment 
    env.render()
    time.sleep(0.015)
    done = terminated or truncated
# Close the environment
file.close()
env.close()

# Run the plotting script
plotting_script_path = 'test_plot.py'
try:
    result = subprocess.run(['python', plotting_script_path], check=True, text=True, capture_output=True)
except subprocess.CalledProcessError as e:
    print("Plotting script error:")
    print(e.stderr)