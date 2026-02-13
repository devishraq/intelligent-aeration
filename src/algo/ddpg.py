import numpy as np
from stable_baselines3 import DDPG
from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise
from ..config.hyperparameters import get_params


def get_agent(env, seed=None, tensorboard_log=None, verbose=0, device="cpu", **kwargs):
    params = get_params("ddpg", kwargs)
    if "action_noise" not in params:
        n = env.action_space.shape[0]
        params["action_noise"] = OrnsteinUhlenbeckActionNoise(np.zeros(n), 12.0 * np.ones(n))
    return DDPG("MlpPolicy", env, seed=seed, tensorboard_log=tensorboard_log,
               verbose=verbose, device=device, **params)
