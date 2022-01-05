from pyvirtualdisplay import Display
display = Display(visible=False, size=(1400,900))
_start = display.start()

import ray
from ray import tune
from ray.rllib.agents.ppo import PPOTrainer
import pybullet_envs

ENV = 'HumanoidBulletEnv-v0'
import gym
from ray.tune.registry import register_env

def make_env(env_config): #creates the virtual environment for the learning simulation
    import pybullet_envs
    return gym.make('HumanoidBulletEnv-v0')
register_env('HumanoidBulletEnv-v0', make_env)
TARGET = 6000
Trainer = PPOTrainer

tune.run(
    Trainer,
    stop={"episodeTargetMean": TARGET},
    config={
        "env" : ENV,
        "num_workers" : 21,
        "num_gpus" : 1,
        "monitor" : True,
        "evaluation_num_episodes" : 100,
        "gamma" : 0.995, #factor for the decision process
        "lambda" : 0.95, #bias and variance trade-off
        "clip_param" : 0.2, #used to prevent the policy from being changed too aggresively, makes it more passive and exponential
        "kl-coeff" : 1.0, #coefficient for the divergebnce penalty, to help prevent PPO from changing too quickly
        "num_sgd_iter" : 20, #number of epochs
        "lr" : 0.0001, #learning rate
        "sgd_minibatch_size" : 32768, #epoch batch size
        "train_batch_size" : 320_000, #number of samples passed in each batch
        "model" : {
            "free_log_std" : True, #makes the second half of the model output bias variables instead of variables that are dependant on the state
        },
        "batch_mode" : "complete_episodes", #choose to roll out full episodes from the workers
        "observation_filter" : "MeanStdFilter", #normalizes observations based on previous states
    }
)