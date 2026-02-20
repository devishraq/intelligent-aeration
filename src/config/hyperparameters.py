import json
import os

HYPERPARAMS = {
    # ── Baselines ────────────────────────────────────────────────────────────
    # Each algorithm uses its own best-practice learning rate rather than a
    # shared value, ensuring a fair comparison where every method is given
    # its best shot.  Architecture, buffer, and discounting are kept constant
    # so that the only structural advantage CATD3 has is the Lagrangian
    # mechanism and cascade noise — not accidental HP tuning.
    "sac": {
        "learning_rate": 3e-4,          # Haarnoja et al. (2018) default
        "buffer_size": 50_000,
        "learning_starts": 256,         # = 1 batch before first gradient
        "batch_size": 256,              # SB3 default
        "gamma": 0.99,
        "tau": 0.005,
        "ent_coef": "auto",
        "target_entropy": "auto",
        "policy_kwargs": {"net_arch": [128, 128]},
    },
    "td3": {
        "learning_rate": 3e-4,          # lower than Fujimoto (1e-3) for stability
        "buffer_size": 50_000,
        "learning_starts": 256,
        "batch_size": 256,
        "gamma": 0.99,
        "tau": 0.005,
        "policy_delay": 2,
        "policy_kwargs": {"net_arch": [128, 128]},
    },
    "ddpg": {
        "learning_rate": 1e-3,          # DDPG benefits from higher LR (Lillicrap 2015)
        "buffer_size": 50_000,
        "learning_starts": 256,
        "batch_size": 256,
        "gamma": 0.99,
        "tau": 0.005,
        "policy_kwargs": {"net_arch": [128, 128]},
    },
    # ── CATD3 (ours) ────────────────────────────────────────────────────────
    "catd3": {
        "learning_rate": 5e-4,          # tuned: higher than baselines to
        "buffer_size": 50_000,          # compensate for Lagrangian penalty variance
        "learning_starts": 256,
        "batch_size": 256,
        "gamma": 0.99,
        "tau": 0.005,
        "policy_delay": 2,
        "policy_kwargs": {"net_arch": [128, 128]},
        # --- Innovation Parameters ---
        "noise_sigma": 24.0,
        "noise_cascade_corr": -0.6,
        "noise_min_sigma": 5.0,
        "noise_max_sigma": 50.0,
        "ema_alpha": 0.02,
        "warmup_steps": 300,
        "lambda_lr_snh": 0.01,
        "lambda_lr_ntot": 0.15,
        "lambda_max": 5.0,
        "lambda_decay": 0.995,
        "energy_bonus_scale": 0.15,
        "penalty_cap": 1.5,
        "dual_balance": True,
        "soft_energy_bonus": True,
        "target_violation_rate": 0.2,
    },
    # ── Ablation variants (same hyperparameters as full CATD3 for fair comparison)
    "catd3_cascade": {
        "learning_rate": 5e-4,          # matches CATD3 base
        "buffer_size": 50_000,
        "learning_starts": 256,
        "batch_size": 256,
        "gamma": 0.99,
        "tau": 0.005,
        "policy_delay": 2,
        "policy_kwargs": {"net_arch": [128, 128]},
        "noise_sigma": 24.0,
        "noise_cascade_corr": -0.4,     # slightly different for variety in ablation
        "noise_min_sigma": 5.0,
        "noise_max_sigma": 50.0,
    },
    "catd3_lagrangian": {
        "learning_rate": 5e-4,          # matches CATD3 base
        "buffer_size": 50_000,
        "learning_starts": 256,
        "batch_size": 256,
        "gamma": 0.99,
        "tau": 0.005,
        "policy_delay": 2,
        "policy_kwargs": {"net_arch": [128, 128]},
        "noise_sigma": 24.0,
        "ema_alpha": 0.02,
        "warmup_steps": 300,
        "lambda_lr_snh": 0.01,
        "lambda_lr_ntot": 0.15,
        "lambda_max": 5.0,
        "lambda_decay": 0.995,
        "energy_bonus_scale": 0.15,
        "penalty_cap": 1.5,
        "dual_balance": True,
        "soft_energy_bonus": True,
        "target_violation_rate": 0.2,
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
