import numpy as np
from stable_baselines3 import TD3
from stable_baselines3.common.noise import NormalActionNoise
from ..config.hyperparameters import get_params


def get_agent(env, seed=None, tensorboard_log=None, verbose=0, device="auto", **kwargs):
    params = get_params("td3", kwargs)

    if "action_noise" not in params:    
        n = env.action_space.shape[0]
        params["action_noise"] = NormalActionNoise(np.zeros(n), 24.0 * np.ones(n))
    
    return TD3("MlpPolicy", env, seed=seed, tensorboard_log=tensorboard_log, verbose=verbose, device=device, **params)
