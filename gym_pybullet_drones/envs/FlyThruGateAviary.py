import os
import numpy as np
import pybullet as p
import pkg_resources
import torch
from gym_pybullet_drones.utils.enums import DroneModel, Physics, ActionType, ObservationType
from gym_pybullet_drones.envs.BaseRLAviary import BaseRLAviary
from gymnasium import spaces
from path_planning import bezier_curve

class FlyThruGateAviary(BaseRLAviary):
    """Single agent RL problem: fly through a gate."""

    ################################################################################

    def __init__(self,
                 drone_model: DroneModel = DroneModel.CF2X,
                 initial_xyzs=None,
                 initial_rpys=None,
                 physics: Physics = Physics.PYB,
                 freq: int = 240,
                 aggregate_phy_steps: int = 12,
                 gui=False,
                 record=False,
                 obs: ObservationType = ObservationType.KIN,
                 act: ActionType = ActionType.PID,
                 episode_len_sec: int = 5,
                 **kwargs):
        """Initialization of a single agent RL environment.

        Using the generic single agent RL superclass.

        Parameters
        ----------
        drone_model : DroneModel, optional
            The desired drone type (detailed in an .urdf file in folder `assets`).
        initial_xyzs: ndarray | None, optional
            (NUM_DRONES, 3)-shaped array containing the initial XYZ position of the drones.
        initial_rpys: ndarray | None, optional
            (NUM_DRONES, 3)-shaped array containing the initial orientations of the drones (in radians).
        physics : Physics, optional
            The desired implementation of PyBullet physics/custom dynamics.
        freq : int, optional
            The frequency (Hz) at which the physics engine steps.
        aggregate_phy_steps : int, optional
            The number of physics steps within one call to `BaseAviary.step()`.
        gui : bool, optional
            Whether to use PyBullet's GUI.
        record : bool, optional
            Whether to save a video of the simulation in folder `files/videos/`.
        obs : ObservationType, optional
            The type of observation space (kinematic information or vision)
        act : ActionType, optional
            The type of action space (1 or 3D; RPMS, thurst and torques, or waypoint with PID control)

        """
        self.DRONE_INIT_LOW_BOUND = torch.tensor([-8.0, -8.0, 0.1])
        self.DRONE_INIT_UP_BOUND = torch.tensor([8.0, 8.0, 4.0])
        self.initial_xyzs = None; self.initial_rpys = None
        if self.initial_xyzs is None:
            self.initial_xyzs = (torch.rand(1, 3) * (self.DRONE_INIT_UP_BOUND - self.DRONE_INIT_LOW_BOUND) + self.DRONE_INIT_LOW_BOUND).numpy()  
        if self.initial_rpys is None:
            self.initial_rpys = (torch.rand(1, 3) * (np.pi / 4) - (np.pi / 8)).numpy()

        super().__init__(drone_model=drone_model,
                         initial_xyzs=self.initial_xyzs,
                         initial_rpys=self.initial_rpys,
                         physics=physics,
                         pyb_freq=freq,
                         ctrl_freq=freq // aggregate_phy_steps,
                         gui=gui,
                         record=record,
                         obs=obs,
                         act=act,
                         **kwargs)
        self.EPISODE_LEN_SEC = episode_len_sec
        self.EPISODE_ALLOWED_BOUNDS = np.array([[-10., -10., 0.05], [10., 10., 10.]])  # x,y,z, z = 0.08 is too close to the ground   

    ################################################################################

    def resetDirectCameraPosition(self, width=640, height=480, fps=24,
                                distance=7, pitch=-30, roll=0,
                                fov=60, nV=0.1, fV=1000.0):
        X, Y, Z, yaw = self.GATE_PARAMS_1[:4]
        self.VID_WIDTH = int(width)
        self.VID_HEIGHT = int(height)
        self.FRAME_PER_SEC = fps
        self.CAPTURE_FREQ = int(self.PYB_FREQ / self.FRAME_PER_SEC)
        self.CAM_VIEW = p.computeViewMatrixFromYawPitchRoll(distance=distance,
                                                            yaw=yaw,
                                                            pitch=pitch,
                                                            roll=roll,
                                                            cameraTargetPosition=[X, Y, Z],
                                                            upAxisIndex=2,
                                                            physicsClientId=self.CLIENT
                                                            )
        self.CAM_PRO = p.computeProjectionMatrixFOV(fov=fov,
                                                    aspect=self.VID_WIDTH / self.VID_HEIGHT,
                                                    nearVal=nV,
                                                    farVal=fV
                                                    )

    ################################################################################

    # def reset(self, *args, **kwargs):
    #     self.GATE_LOW_BOUND = [-4.0, -4.0, 0.75, -np.pi, 1.57]
    #     self.GATE_UP_BOUND = [4.0, 4.0, 4.0, np.pi, 1.57]
    #     self.GATE_PARAMS = np.random.uniform(self.GATE_LOW_BOUND, self.GATE_UP_BOUND)
    #     self.DRONE_INIT_LOW_BOUND = [-4.0, -4.0, 0.1]
    #     self.DRONE_INIT_UP_BOUND = [4.0, 4.0, 4.0]
    #     self.INIT_XYZS = np.random.uniform(self.DRONE_INIT_LOW_BOUND, self.DRONE_INIT_UP_BOUND, (self.NUM_DRONES, 3)) 
    #     self.INIT_RPYS = np.random.uniform(-np.pi / 8, np.pi / 8, (self.NUM_DRONES, 3))
    #     self.gate_target = 0
    #     self.resetDirectCameraPosition()

        # return super().reset(*args, **kwargs)
    
    # def reset(self, *args, **kwargs):
    #     self.GATE_LOW_BOUND = torch.tensor([-4.0, -4.0, 0.75, -np.pi, 1.57])
    #     self.GATE_UP_BOUND = torch.tensor([4.0, 4.0, 4.0, np.pi, 1.57])
    #     self.GATE_PARAMS = torch.rand(5) * (self.GATE_UP_BOUND - self.GATE_LOW_BOUND) + self.GATE_LOW_BOUND
    #     self.GATE_PARAMS = self.GATE_PARAMS.numpy()
        
    #     # Define the limits for X, Y, and Z (for movement range)
    #     x_min, x_max = -0.01, 0.01
    #     y_min, y_max = -0.01, 0.01
    #     z_min, z_max =  0.05, 0.01
    #     self.MOVEMENT_DIRECTION = torch.rand(3) * torch.tensor([x_max - x_min, y_max - y_min, z_max - z_min]) + torch.tensor([x_min, y_min, z_min])
    #     self.MOVEMENT_DIRECTION = self.MOVEMENT_DIRECTION.numpy()
        

    #     self.DRONE_INIT_LOW_BOUND = torch.tensor([-4.0, -4.0, 0.1])
    #     self.DRONE_INIT_UP_BOUND = torch.tensor([4.0, 4.0, 4.0])
    #     self.INIT_XYZS = torch.rand(self.NUM_DRONES, 3) * (self.DRONE_INIT_UP_BOUND - self.DRONE_INIT_LOW_BOUND) + self.DRONE_INIT_LOW_BOUND
    #     self.INIT_RPYS = torch.rand(self.NUM_DRONES, 3) * (np.pi / 4) - (np.pi / 8)

    #     self.gate_target = 0
    #     self.resetDirectCameraPosition()

    #     return super().reset(*args, **kwargs)

    def reset(self, *args, **kwargs):
        # Set bounds for gate positions
        self.GATE_LOW_BOUND = torch.tensor([-8.0, -8.0, 0.3, -np.pi, 1.57])
        self.GATE_UP_BOUND = torch.tensor([8.0, 8.0, 4.0, np.pi, 1.57])

        # Generate random parameters for the first gate
        self.GATE_PARAMS_1 = torch.rand(5) * (self.GATE_UP_BOUND - self.GATE_LOW_BOUND) + self.GATE_LOW_BOUND
        self.GATE_PARAMS_1 = self.GATE_PARAMS_1.numpy()

        # Generate random parameters for the second gate
        # self.GATE_PARAMS_2 = torch.rand(5) * (self.GATE_UP_BOUND - self.GATE_LOW_BOUND) + self.GATE_LOW_BOUND
        # self.GATE_PARAMS_2 = self.GATE_PARAMS_2.numpy()

        # Randomize drone position
        self.initial_xyzs = (torch.rand(1, 3) * (self.DRONE_INIT_UP_BOUND - self.DRONE_INIT_LOW_BOUND) + self.DRONE_INIT_LOW_BOUND).numpy()
        self.initial_rpys = (torch.rand(1, 3) * (np.pi / 4) - (np.pi / 8)).numpy()
        self.INIT_XYZS = self.initial_xyzs
        self.INIT_RPYS = self.initial_rpys

        # Randomize obstacle position
        line_vector = self.GATE_PARAMS_1[:3] - self.initial_xyzs[0]
        longitudinal_factor = (torch.rand(1) * 0.6 + 0.2).numpy()
        obstacle_position = self.initial_xyzs[0] + (longitudinal_factor * line_vector)
        lateral_vector = np.cross(line_vector, np.array([0.0, 0.0, 1.0]))
        lateral_vector /= np.linalg.norm(lateral_vector)
        lateral_offset = lateral_vector * ((torch.rand(1) * 2 - 1)).numpy()
        obstacle_position += lateral_offset
        obstacle_position[2] = (torch.rand(1) * 4).numpy()
        self.OBSTACLE_PARAMS = obstacle_position
        self.OBSTACLE_RADIUS = 0.25

        # # Randomize Gate 2 in proximity of Gate 1
        # gate1_x, gate1_y, gate1_z = self.GATE_PARAMS_1[:3]
        # drone_x, drone_y, drone_z = self.initial_xyzs[0]
        # approach_vector_x = gate1_x - drone_x
        # approach_vector_y = gate1_y - drone_y
        # approach_vector_length = np.sqrt(approach_vector_x**2 + approach_vector_y**2)
        # approach_dir_x = approach_vector_x / approach_vector_length
        # approach_dir_y = approach_vector_y / approach_vector_length
        # max_distance = 1.0
        # off_x = np.random.uniform(0, max_distance)
        # off_y = np.random.uniform(0, max_distance)
        # off_z = np.random.uniform(0, max_distance)
        # gate2_x = gate1_x + off_x * (approach_dir_x)
        # gate2_y = gate1_y + off_y * (approach_dir_y)
        # gate2_z = gate1_z + off_z
        # gate2_yaw = 0 #np.random.uniform(-np.pi, np.pi)
        # self.GATE_PARAMS_2 = np.array([gate2_x, gate2_y, gate2_z, gate2_yaw, 1.0])

        # # Set the first gate as the initial target
        # self.current_gate_target = 1
        # self.reached_first_gate = False
        # self.prev_state = None

        # Reset the camera position
        self.resetDirectCameraPosition()

        return super().reset(*args, **kwargs)

    ############################################################################

    # def step(self, *args, **kwargs):
        # """
        # Extends the superclass method and move the gate in a direction at each step.
        # Also ensures that GATE_PARAMS does not exceed the specified bounds.
        # """
        # # Update the GATE_PARAMS with the movement direction
        # new_gate_params = self.GATE_PARAMS[:3] + self.MOVEMENT_DIRECTION
        # # Check if the new GATE_PARAMS exceed the bounds
        # for i in range(3):  
        #     if new_gate_params[i] < self.GATE_LOW_BOUND[i]:
        #         new_gate_params[i] = self.GATE_LOW_BOUND[i]
        #     elif new_gate_params[i] > self.GATE_UP_BOUND[i]:
        #         new_gate_params[i] = self.GATE_UP_BOUND[i]
        # self.GATE_PARAMS[:3] = [0,0,0] #new_gate_params

        # state = self._getDroneStateVector(0)
        # drone_position = state[0:3].reshape(3)
        # gate_1_center = self.GATE_PARAMS_1[:3]
        # distance_to_gate_1 = np.linalg.norm(gate_1_center - drone_position)

        # if self.current_gate_target == 1 and distance_to_gate_1 < 0.1:  # Threshold for reaching the gate
        #     self.current_gate_target = 2  # Switch to the second gate target

        # return super().step(*args, **kwargs)

    ############################################################################

    def _addObstacles(self):
        """Add obstacles to the environment.

        Extends the superclass method and add the gate build of cubes and an architrave.

        """
        super()._addObstacles()

        def make_gate(x, y, z, yaw, size=1.0):
        #Compute left bottom corner from center and yaw
            x = x - 0.5 * size * np.cos(yaw)
            y = y - 0.5 * size * np.sin(yaw)
            z = z - 0.5 * size

            p.loadURDF(pkg_resources.resource_filename('gym_pybullet_drones', 'assets/gate.urdf'),
                       [x, y, z], # left bottom corner
                       p.getQuaternionFromEuler([0, 0, yaw]),
                       physicsClientId=self.CLIENT,
                       globalScaling=size
                       )

        # print("OBSTACLE: ", self.GATE_PARAMS)
        make_gate(*self.GATE_PARAMS_1)

    ################################################################################

    # def _actionSpace(self):
    #     """Extends the action space of the base class by adding yaw control."""
    #     # Call the parent class method to get the original action space
    #     base_action_space = super()._actionSpace()

    #     # Retrieve the existing bounds and add the yaw control
    #     low = np.hstack([base_action_space.low, np.array([-np.pi for _ in range(self.NUM_DRONES)]).reshape(-1, 1)])
    #     high = np.hstack([base_action_space.high, np.array([np.pi for _ in range(self.NUM_DRONES)]).reshape(-1, 1)])

    #     # Return the extended action space
    #     return spaces.Box(low=low, high=high, dtype=np.float32)
    
    ################################################################################

    def _observationSpace(self):
        """Returns the observation space.

        Returns
        -------
        gym.spaces.Box
            The observation space.

        """
        obs_space = super()._observationSpace()
        if self.OBS_TYPE == ObservationType.KIN:
            low_bound = np.array([self.GATE_LOW_BOUND for j in range(self.NUM_DRONES)])
            up_bound = np.array([self.GATE_UP_BOUND for j in range(self.NUM_DRONES)])
            obs_lower_bound = np.hstack([obs_space.low.reshape(self.NUM_DRONES, -1), low_bound])
            obs_upper_bound = np.hstack([obs_space.high.reshape(self.NUM_DRONES, -1), up_bound])
            return spaces.Box(low=obs_lower_bound, high=obs_upper_bound, dtype=np.float32)
        else:
            return obs_space
        
    # def _observationSpace(self):
    #     """Returns the observation space including both gates at all times."""
    #     obs_space = super()._observationSpace()
    #     if self.OBS_TYPE == ObservationType.KIN:
    #         low_bound = np.array([self.GATE_LOW_BOUND for _ in range(self.NUM_DRONES)])
    #         up_bound = np.array([self.GATE_UP_BOUND for _ in range(self.NUM_DRONES)])
    #         obs_lower_bound = np.hstack([obs_space.low.reshape(self.NUM_DRONES, -1), low_bound, low_bound])
    #         obs_upper_bound = np.hstack([obs_space.high.reshape(self.NUM_DRONES, -1), up_bound, up_bound])
    #         return spaces.Box(low=obs_lower_bound, high=obs_upper_bound, dtype=np.float32)
    #     else:
    #         return obs_space


    ################################################################################

    def _computeObs(self):
        obs = super()._computeObs()
        if self.OBS_TYPE == ObservationType.KIN:
            self.gate_poses_1 = np.array([self.GATE_PARAMS_1]).reshape(-1)
            obs = np.hstack([obs,
                            np.tile(self.gate_poses_1, (self.NUM_DRONES, 1))
                            ])
            return obs
        else:
            return obs

    # def _computeObs(self):
    #     obs = super()._computeObs()
    #     if self.OBS_TYPE == ObservationType.KIN:
    #         self.gate_poses_1 = np.array([self.GATE_PARAMS_1]).reshape(-1)
    #         self.gate_poses_2 = np.array([self.GATE_PARAMS_2]).reshape(-1)
    #         obs = np.hstack([obs,
    #                         np.tile(self.gate_poses_1, (self.NUM_DRONES, 1)),
    #                         np.tile(self.gate_poses_2, (self.NUM_DRONES, 1))
    #                         ])
    #         return obs
    #     else:
    #         return obs

    ################################################################################

    def _computeReward(self):
        """Computes the current reward value.

        Returns
        -------
        float
            The reward.

        # """
        state = self._getDroneStateVector(0)
        drone_position = state[0:3].reshape(3)
        drone_velocity = state[10:13].reshape(3)
        gate_center = self.GATE_PARAMS_1[:3]
        drone_start = self.initial_xyzs[0]  
        obstacle_position = self.OBSTACLE_PARAMS.reshape(3)
        desired_direction = gate_center - drone_position

        # Reward for proximity to the gate
        distance_to_gate = np.linalg.norm(desired_direction)
        max_distance = np.linalg.norm(gate_center - drone_start)
        # max_distance = np.linalg.norm(gate_center - drone_start)
        # Normalize the distance (clamp to [0, 1])
        # normalized_distance = np.clip(distance_to_gate / max_distance, 0.0, 1.0)
        # Apply a non-linear reward curve (quadratic or exponential decay)
        # reward = 1 - (normalized_distance ** 2)  # Quadratic decay
        # reward = np.exp(-normalized_distance * 4)  # Exponential decay with steepness factor of 4
        proximity_reward = 1 / ((distance_to_gate)+0.001)  #1 - norm_distance_to_gate  

        reward = proximity_reward

        return reward
    
    # def _computeReward(self):
    #     """Computes the reward for the current state of the drone.
        
    #     Returns
    #     -------
    #     float
    #         The computed reward.
    #     """
    #     # Step 1: Calculate progress reward (rp)
        
    #     # Get drone's current position
    #     state = self._getDroneStateVector(0)
    #     drone_position = state[0:3]  # (x, y, z)

    #     # Define Gate 1 and Gate 2 positions
    #     gate1_position = self.GATE_PARAMS_1[:3]
    #     gate2_position = self.GATE_PARAMS_2[:3]

    #     # Calculate distances to Gate 1 and Gate 2 in x-y plane
    #     dist_to_gate1 = np.linalg.norm(drone_position[:2] - gate1_position[:2])
    #     dist_to_gate2 = np.linalg.norm(drone_position[:2] - gate2_position[:2])
    #     dist_g1_g2 = np.linalg.norm(gate1_position[:2] - gate2_position[:2])

    #     # Calculate distance in x-y-z plane
    #     dist_to_gate2_prox = np.linalg.norm(drone_position - gate2_position)

    #     # Calculate max distance to Gate 2 from initial pos
    #     max_dist_g2 = np.linalg.norm(self.initial_xyzs[0] - gate2_position)

    #     # Cross track angle
    #     a = dist_to_gate1; b = dist_to_gate2; c = dist_g1_g2
    #     phi = np.arccos((b**2 + c**2 - a**2)/(2*b*c)) 

    #     # normalized proximity reward
    #     prox_reward = 1 - (dist_to_gate2_prox/max_dist_g2)

    #     # normalized cross-track reward
    #     cross_t_reward = 1 - (phi/np.pi)

    #     reward = prox_reward + 0.5*cross_t_reward
 
    #     return reward

    ################################################################################

    def _computeTerminated(self):
        """Computes the current done value.

        Returns
        -------
        bool
            Whether the current episode is done.

        """
        state = self._getDroneStateVector(0)
        if np.linalg.norm(self.GATE_PARAMS_1[:3]-state[0:3].reshape(3)) <= .01: 
            return True
        else:
            return False

    def _computeTruncated(self): 
        """Computes the current truncated value.

        Returns
        -------
        bool
            Whether the current episode is truncated.

        """
        # state = self._getDroneStateVector(0)
        # drone_position = state[0:3].reshape(3)
        # # check that is it inside the allowed bounds
        # if np.any(drone_position < self.EPISODE_ALLOWED_BOUNDS[0]) or np.any(drone_position > self.EPISODE_ALLOWED_BOUNDS[1]):
        #     return True
        state = self._getDroneStateVector(0)
        if self.step_counter/self.PYB_FREQ > self.EPISODE_LEN_SEC or np.linalg.norm(self.GATE_PARAMS_1[:3]-state[0:3].reshape(3)) <= .01:
            return True
        else:
            return False

    ################################################################################

    def _computeInfo(self):
        """Computes the current info dict(s).

        Unused.

        Returns
        -------
        dict[str, int]
            Dummy value.

        """
        return {}

    ################################################################################

    # def _clipAndNormalizeState(self,
    #                            state
    #                            ):
    #     """Normalizes a drone's state to the [-1,1] range.
    #
    #     Parameters
    #     ----------
    #     state : ndarray
    #         (20,)-shaped array of floats containing the non-normalized state of a single drone.
    #
    #     Returns
    #     -------
    #     ndarray
    #         (20,)-shaped array of floats containing the normalized state of a single drone.
    #
    #     """
    #     MAX_LIN_VEL_XY = 3
    #     MAX_LIN_VEL_Z = 1
    #
    #     MAX_XY = MAX_LIN_VEL_XY * self.EPISODE_LEN_SEC
    #     MAX_Z = MAX_LIN_VEL_Z * self.EPISODE_LEN_SEC
    #
    #     MAX_PITCH_ROLL = np.pi  # Full range
    #
    #     clipped_pos_xy = np.clip(state[0:2], -MAX_XY, MAX_XY)
    #     clipped_pos_z = np.clip(state[2], 0, MAX_Z)
    #     clipped_rp = np.clip(state[7:9], -MAX_PITCH_ROLL, MAX_PITCH_ROLL)
    #     clipped_vel_xy = np.clip(state[10:12], -MAX_LIN_VEL_XY, MAX_LIN_VEL_XY)
    #     clipped_vel_z = np.clip(state[12], -MAX_LIN_VEL_Z, MAX_LIN_VEL_Z)
    #
    #     if self.GUI:
    #         self._clipAndNormalizeStateWarning(state,
    #                                            clipped_pos_xy,
    #                                            clipped_pos_z,
    #                                            clipped_rp,
    #                                            clipped_vel_xy,
    #                                            clipped_vel_z
    #                                            )
    #
    #     normalized_pos_xy = clipped_pos_xy / MAX_XY
    #     normalized_pos_z = clipped_pos_z / MAX_Z
    #     normalized_rp = clipped_rp / MAX_PITCH_ROLL
    #     normalized_y = state[9] / np.pi  # No reason to clip
    #     normalized_vel_xy = clipped_vel_xy / MAX_LIN_VEL_XY
    #     normalized_vel_z = clipped_vel_z / MAX_LIN_VEL_XY
    #     normalized_ang_vel = state[13:16] / np.linalg.norm(state[13:16]) if np.linalg.norm(
    #         state[13:16]) != 0 else state[13:16]
    #
    #     norm_and_clipped = np.hstack([normalized_pos_xy,
    #                                   normalized_pos_z,
    #                                   state[3:7],
    #                                   normalized_rp,
    #                                   normalized_y,
    #                                   normalized_vel_xy,
    #                                   normalized_vel_z,
    #                                   normalized_ang_vel,
    #                                   state[16:20]
    #                                   ]).reshape(20, )
    #
    #     return norm_and_clipped
    #
    # ################################################################################
    #
    # def _clipAndNormalizeStateWarning(self,
    #                                   state,
    #                                   clipped_pos_xy,
    #                                   clipped_pos_z,
    #                                   clipped_rp,
    #                                   clipped_vel_xy,
    #                                   clipped_vel_z,
    #                                   ):
    #     """Debugging printouts associated to `_clipAndNormalizeState`.
    #
    #     Print a warning if values in a state vector is out of the clipping range.
    #
    #     """
    #     if not (clipped_pos_xy == np.array(state[0:2])).all():
    #         print("[WARNING] it", self.step_counter,
    #               "in FlyThruGateAviary._clipAndNormalizeState(), clipped xy position [{:.2f} {:.2f}]".format(state[0],
    #                                                                                                           state[1]))
    #     if not (clipped_pos_z == np.array(state[2])).all():
    #         print("[WARNING] it", self.step_counter,
    #               "in FlyThruGateAviary._clipAndNormalizeState(), clipped z position [{:.2f}]".format(state[2]))
    #     if not (clipped_rp == np.array(state[7:9])).all():
    #         print("[WARNING] it", self.step_counter,
    #               "in FlyThruGateAviary._clipAndNormalizeState(), clipped roll/pitch [{:.2f} {:.2f}]".format(state[7],
    #                                                                                                          state[8]))
    #     if not (clipped_vel_xy == np.array(state[10:12])).all():
    #         print("[WARNING] it", self.step_counter,
    #               "in FlyThruGateAviary._clipAndNormalizeState(), clipped xy velocity [{:.2f} {:.2f}]".format(state[10],
    #                                                                                                           state[
    #                                                                                                               11]))
    #     if not (clipped_vel_z == np.array(state[12])).all():
    #         print("[WARNING] it", self.step_counter,
    #               "in FlyThruGateAviary._clipAndNormalizeState(), clipped z velocity [{:.2f}]".format(state[12]))

