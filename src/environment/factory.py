import os
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv

from .bsm1 import BSM1Env
from .rewards import ZAMORRewardCalculator


def create_training_env(method, phase_config, seed=42, n_envs=None, normalize=True, norm_obs=True, norm_reward=False, clip_obs=10.0, use_subproc=True):
    if n_envs is None:
        n_envs = 1

    reward_calc = ZAMORRewardCalculator()

    def make_env():
        env = BSM1Env(scenario=phase_config.scenario, episode_days=phase_config.episode_days, noise_std=phase_config.noise_std,seed=seed, reward_calculator=reward_calc,)
        return Monitor(env)

    vec_env = make_vec_env(make_env, n_envs=n_envs, seed=seed, vec_env_cls=DummyVecEnv)

    if normalize:
        vec_env = VecNormalize(vec_env,norm_obs=False,norm_reward=False,clip_obs=10.0,clip_reward=10.0,epsilon=1e-8,)

    return vec_env
