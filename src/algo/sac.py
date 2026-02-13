from stable_baselines3 import SAC
from ..config.hyperparameters import get_params


def get_agent(env, seed=None, tensorboard_log=None, verbose=0, device="auto", **kwargs):

    return SAC("MlpPolicy", env, seed=seed, tensorboard_log=tensorboard_log, verbose=verbose, device=device, **get_params("sac", kwargs))
