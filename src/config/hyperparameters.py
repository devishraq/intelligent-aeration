import json
import os

HYPERPARAMS = {
    "sac": {
        "learning_rate": 3e-4,
        "buffer_size": 50_000,
        "learning_starts": 500,
        "batch_size": 128*4,
        "gamma": 0.99,
        "tau": 0.005,
        "ent_coef": "auto",
        "policy_kwargs": {"net_arch": [128, 128]},
    },
    "td3": {
        "learning_rate": 1e-3,
        "buffer_size": 50_000,
        "learning_starts": 500,
        "batch_size": 128*4,
        "gamma": 0.99,
        "tau": 0.005,
        "policy_delay": 2,
        "policy_kwargs": {"net_arch": [128, 128]},
    },
    "ddpg": {
        "learning_rate": 1e-3,
        "buffer_size": 50_000,
        "learning_starts": 500,
        "batch_size": 128*4,
        "gamma": 0.99,
        "tau": 0.005,
        "policy_kwargs": {"net_arch": [128, 128]},
    },
    "catd3": {
        "learning_rate": 1e-3,
        "buffer_size": 50_000,
        "learning_starts": 500,
        "batch_size": 128*4,
        "gamma": 0.99,
        "tau": 0.005,
        "policy_delay": 2,
        "policy_kwargs": {"net_arch": [128, 128]},
    },
}


def get_params(algo_name, overrides=None, config_file=None):
    algo_name = algo_name.lower().strip()
    if algo_name not in HYPERPARAMS:
        raise ValueError(f"Unknown algorithm '{algo_name}'. Available: {list(HYPERPARAMS.keys())}")
    config = HYPERPARAMS[algo_name].copy()
    if config_file and os.path.exists(config_file):
        with open(config_file) as f:
            config.update(json.load(f))
    if overrides:
        config.update(overrides)
    return config
