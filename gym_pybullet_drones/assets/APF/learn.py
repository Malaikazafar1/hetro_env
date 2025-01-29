"""Script demonstrating the use of `gym_pybullet_drones`'s Gymnasium interface.

Classes HoverAviary and MultiHoverAviary are used as learning envs for the PPO algorithm.

Example
-------
In a terminal, run as:

    $ python learn.py --multiagent false
    $ python learn.py --multiagent true

Notes
-----
This is a minimal working example integrating `gym-pybullet-drones` with 
reinforcement learning library `stable-baselines3`.

$ tensorboard --logdir ./results/save-<env>-<algo>-<obs>-<act>-<time-date>/tb/

"""
import os
import time
from datetime import datetime
import argparse
import json
from pathlib import Path
import gymnasium as gym
import inspect
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.policies import ActorCriticPolicy
from sb3_contrib import TQC

from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.envs.DynamicAgent import DynamicAgent
from gym_pybullet_drones.utils.utils import sync, str2bool
from gym_pybullet_drones.utils.enums import ObservationType, ActionType, DroneModel

DEFAULT_GUI = False
DEFAULT_RECORD_VIDEO = True
DEFAULT_OUTPUT_FOLDER = 'results'
DEFAULT_COLAB = False

DEFAULT_OBS = ObservationType('kin') # 'kin' or 'rgb'
DEFAULT_ACT = ActionType('pid') # 'rpm' or 'pid' or 'vel' or 'one_d_rpm' or 'one_d_pid'
DEFAULT_AGENTS = 1
DEFAULT_MA = False
DEFAULT_DRONE_MODEL = DroneModel.RACE

AGGR_PHY_STEPS = 5
EPISODE_LEN_SEC = 12
TOTAL_STEPS = 50000000#0
N_ENVS = 8 #16
BATCH_SIZE = 256*2 #128 #8200
REWARD_EARLY_STOPPING = 1e8
N_STEPS = 2048 #512
ENT_COEFF = 0.01 #0.01
GAMMA = 0.99   #0.99
CLIP_RANGE = 0.2 #0.2
ALG = "PPO"


def run(alg=ALG, multiagent=DEFAULT_MA, output_folder=DEFAULT_OUTPUT_FOLDER, gui=DEFAULT_GUI, plot=True, colab=DEFAULT_COLAB, record_video=DEFAULT_RECORD_VIDEO, local=True):

    output_folder = os.path.join(output_folder, f'{alg}-{datetime.now().strftime("%m.%d.%Y_%H.%M.%S")}')
    if not os.path.exists(output_folder):
        os.makedirs(output_folder+'/')

    params = {"DEFAULT_OBS": str(DEFAULT_OBS), "DEFAULT_ACT": str(DEFAULT_ACT),
              "DEFAULT_AGENTS": DEFAULT_AGENTS, "DEFAULT_MA": DEFAULT_MA,
              "TOTAL_STEPS": TOTAL_STEPS, "N_ENVS": N_ENVS,
              "N_STEPS": N_STEPS, "BATCH_SIZE": BATCH_SIZE, "REWARD_EARLY_STOPPING": REWARD_EARLY_STOPPING,
              "ALG": ALG, "AGGR_PHY_STEPS": AGGR_PHY_STEPS, "EPISODE_LEN_SEC": EPISODE_LEN_SEC, "ENT_COEFF": ENT_COEFF,
              "GAMMA": GAMMA, "CLIP_RANGE": CLIP_RANGE}
    with open(os.path.join(output_folder, "params.json"), "w") as f:
        json.dump(params, f, indent=4)
    with open(os.path.join(output_folder, "reward_function_src.py"), "w") as f:
        f.write(inspect.getsource(DynamicAgent._computeReward))

    env_kwargs = dict(aggregate_phy_steps=AGGR_PHY_STEPS, obs=DEFAULT_OBS,
                      act=DEFAULT_ACT, episode_len_sec=EPISODE_LEN_SEC,
                      output_folder=output_folder, drone_model = DEFAULT_DRONE_MODEL)


    train_env = make_vec_env(DynamicAgent,
                             env_kwargs=env_kwargs,
                             n_envs=N_ENVS,
                             seed=0
                             )
    eval_env = make_vec_env(DynamicAgent,
                            env_kwargs=env_kwargs,
                            n_envs=1,
                            seed=0
                            )

    # env_kwargs = dict(obs=DEFAULT_OBS,
    #                   act=DEFAULT_ACT)
    # train_env = make_vec_env(HoverAviary,
    #                          env_kwargs=env_kwargs,
    #                          n_envs=N_ENVS,
    #                          seed=0
    #                          )
    # eval_env = make_vec_env(HoverAviary,
    #                         env_kwargs=env_kwargs,
    #                         n_envs=1,
    #                         seed=0
    #                         )

    #### Check the environment's spaces ########################
    print('[INFO] Action space:', train_env.action_space)
    print('[INFO] Observation space:', train_env.observation_space)

    #### Train the model #######################################
    policy_kwargs = dict(activation_fn=torch.nn.ReLU,
                        #  net_arch=[512, 512, dict(vf=[256, 128], pi=[256, 128])]
                         net_arch=[512, 512, 256, 128]
                         )  # or None

    if alg == "PPO":
        model = PPO(ActorCriticPolicy,
                    train_env,
                    policy_kwargs=policy_kwargs,
                    learning_rate=1e-4,
                    gamma=GAMMA,
                    n_steps=N_STEPS,
                    batch_size=BATCH_SIZE,
                    ent_coef=ENT_COEFF,
                    clip_range=CLIP_RANGE,
                    tensorboard_log=output_folder+'/tb/',
                    normalize_advantage=False,
                    verbose=1)
    elif alg == "TQC":
        policy_kwargs = {**policy_kwargs, "n_critics": 2, "n_quantiles": 25}
        model = TQC("MlpPolicy",
                    train_env,
                    top_quantiles_to_drop_per_net=2,
                    policy_kwargs=policy_kwargs,
                    # gamma=0.99,
                    learning_rate=1e-4,
                    train_freq=N_STEPS,
                    buffer_size=100000,
                    batch_size=BATCH_SIZE,
                    # tensorboard_log=filename+'/tb/',
                    verbose=1)

    #### Target cumulative rewards (problem-dependent) ##########
    callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=REWARD_EARLY_STOPPING,
                                                     verbose=1)
    eval_callback = EvalCallback(eval_env,
                                 callback_on_new_best=callback_on_best,
                                 verbose=1,
                                 best_model_save_path=output_folder+'/',
                                 log_path=output_folder+'/',
                                 eval_freq=int(1000),
                                 deterministic=True,
                                 render=False)

    #### Show (and record a video of) the model's performance ##
    test_env = DynamicAgent(record=record_video, **env_kwargs)
    # test_env = HoverAviary(record=record_video, **env_kwargs)

    logger = Logger(logging_freq_hz=int(test_env.CTRL_FREQ),
                    num_drones=DEFAULT_AGENTS if multiagent else 1,
                    output_folder=output_folder,
                    colab=colab
                    )

    def make_mp4():
        dirs = list(Path(output_folder).glob("recording_*"))
        for d in dirs:
            if not d.is_dir():
                continue
            # if len(list(d.glob("*"))) > 0:
            os.system(
                f"ffmpeg -framerate 24 -i {str(d)}/frame_%d.png -vcodec mpeg4 {output_folder}/{TOTAL_STEPS}_{d.parts[-1]}.mp4 > /dev/null 2>&1; rm -r {str(d)}  > /dev/null 2>&1")

    total_steps = TOTAL_STEPS // 10
    # total_steps * (i + 1)

    # for i in range(10):
    model.set_env(train_env)
    model.learn(total_timesteps=TOTAL_STEPS,
                callback=eval_callback,
                log_interval=100)

    #### Save the model ########################################
    model.save(output_folder+'/last_model.zip')

    if os.path.isfile(output_folder+'/best_model.zip'):
        path = output_folder+'/best_model.zip'
    else:
        path = output_folder + '/last_model.zip'

    if alg == "PPO":
        eval_model = PPO.load(path)
    elif alg == "TQC":
        eval_model = TQC.load(path)

    mean_reward, std_reward = evaluate_policy(eval_model,
                                                test_env,
                                                n_eval_episodes=5)
    print("\n\n\nMean reward ", mean_reward, " +- ", std_reward, "\n\n")
    make_mp4()


    #### Print training progression ############################
    if os.path.isfile(output_folder+'/evaluations.npz'):
        with np.load(output_folder+'/evaluations.npz') as data:
            for j in range(data['timesteps'].shape[0]):
                print(str(data['timesteps'][j])+","+str(data['results'][j][0]))


if __name__ == '__main__':
    #### Define and parse (optional) arguments for the script ##
    parser = argparse.ArgumentParser(description='Single agent reinforcement learning example script')
    parser.add_argument('--multiagent',         default=DEFAULT_MA,            type=str2bool,      help='Whether to use example LeaderFollower instead of Hover (default: False)', metavar='')
    parser.add_argument('--gui',                default=DEFAULT_GUI,           type=str2bool,      help='Whether to use PyBullet GUI (default: True)', metavar='')
    parser.add_argument('--record_video',       default=DEFAULT_RECORD_VIDEO,  type=str2bool,      help='Whether to record a video (default: False)', metavar='')
    parser.add_argument('--output_folder',      default=DEFAULT_OUTPUT_FOLDER, type=str,           help='Folder where to save logs (default: "results")', metavar='')
    parser.add_argument('--colab',              default=DEFAULT_COLAB,         type=bool,          help='Whether example is being run by a notebook (default: "False")', metavar='')
    ARGS = parser.parse_args()

    run(**vars(ARGS))
