LIMIT_SNH = 4.0
LIMIT_NTOT = 18.0

LIMITS = {
    "SNH": LIMIT_SNH,
    "Ntot": LIMIT_NTOT,
    "TSS": 30.0,
    "COD": 100.0,
    "BOD5": 10.0,
}

SCENARIOS = ["dry", "rain", "storm"]


class PhaseConfig:
    def __init__(self, name, scenario, noise_std, action_delay, episode_days, max_episodes):
        self.name = name
        self.scenario = scenario
        self.noise_std = noise_std
        self.action_delay = action_delay
        self.episode_days = episode_days
        self.max_episodes = max_episodes


def get_curriculum(mode="demo"):
    if mode == "demo":
        return [
            PhaseConfig("dry_demo", "dry", 0.0, 0, 2.0, 150),
        ]
    return [
        PhaseConfig("phase1_dry", "dry", 0.0, 0, 2.0, 2000),
        PhaseConfig("phase2_rain", "rain", 0.05, 0, 2.0, 1200),
        PhaseConfig("phase3_storm", "storm", 0.1, 2, 2.0, 1200),
    ]


from .hyperparameters import HYPERPARAMS, get_params
