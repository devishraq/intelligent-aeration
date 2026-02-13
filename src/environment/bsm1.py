import gymnasium as gym
import numpy as np
from pathlib import Path
from gymnasium import spaces
from bsm2_python import BSM1OL
from bsm2_python.bsm1_base import SNH, SNO, SO, SALK, Q

from ..config import LIMIT_SNH, LIMIT_NTOT
from .rewards import RewardCalculator

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"

SCENARIOS = {
    "dry": "inf_dry_2006.csv",
    "rain": "inf_rain_2006.csv",
    "storm": "inf_strm_2006.csv",
}

_STABLE_CACHE = {}

# caching for stable snapshots
def _snapshot_bsm(bsm):
    return {
        "r1_y0": bsm.reactor1.y0.copy(),
        "r2_y0": bsm.reactor2.y0.copy(),
        "r3_y0": bsm.reactor3.y0.copy(),
        "r4_y0": bsm.reactor4.y0.copy(),
        "r5_y0": bsm.reactor5.y0.copy(),
        "settler_ys0": bsm.settler.ys0.copy(),
        "y_in1": bsm.y_in1.copy(),
        "y_out1": bsm.y_out1.copy(),
        "y_out2": bsm.y_out2.copy(),
        "y_out3": bsm.y_out3.copy(),
        "y_out4": bsm.y_out4.copy(),
        "y_out5": bsm.y_out5.copy(),
        "y_out5_r": bsm.y_out5_r.copy(),
        "ys_in": bsm.ys_in.copy(),
        "ys_out": bsm.ys_out.copy(),
        "ys_eff": bsm.ys_eff.copy(),
        "klas": bsm.klas.copy(),
        "qintr": float(bsm.qintr),
        "sludge_height": float(bsm.sludge_height),
        "ys_tss_internal": bsm.ys_tss_internal.copy(),
        "ae": float(bsm.ae),
        "pe": float(bsm.pe),
        "me": float(bsm.me),
    }

def _restore_bsm(bsm, snap):
    bsm.reactor1.y0[:] = snap["r1_y0"]
    bsm.reactor2.y0[:] = snap["r2_y0"]
    bsm.reactor3.y0[:] = snap["r3_y0"]
    bsm.reactor4.y0[:] = snap["r4_y0"]
    bsm.reactor5.y0[:] = snap["r5_y0"]
    bsm.settler.ys0[:] = snap["settler_ys0"]
    bsm.y_in1[:] = snap["y_in1"]
    bsm.y_out1[:] = snap["y_out1"]
    bsm.y_out2[:] = snap["y_out2"]
    bsm.y_out3[:] = snap["y_out3"]
    bsm.y_out4[:] = snap["y_out4"]
    bsm.y_out5[:] = snap["y_out5"]
    bsm.y_out5_r[:] = snap["y_out5_r"]
    bsm.ys_in[:] = snap["ys_in"]
    bsm.ys_out[:] = snap["ys_out"]
    bsm.ys_eff[:] = snap["ys_eff"]
    bsm.klas[:] = snap["klas"]
    bsm.qintr = snap["qintr"]
    bsm.sludge_height = snap["sludge_height"]
    bsm.ys_tss_internal = snap["ys_tss_internal"].copy()
    bsm.ae = snap["ae"]
    bsm.pe = snap["pe"]
    bsm.me = snap["me"]
    bsm.stabilized = True

def _get_stable_snapshot(data_path):
    if data_path not in _STABLE_CACHE:
        bsm = BSM1OL(data_in=data_path)
        bsm.stabilize()
        _STABLE_CACHE[data_path] = _snapshot_bsm(bsm)

    return _STABLE_CACHE[data_path]


class BSM1Env(gym.Env):
    metadata = {"render_modes": []}

    _OBS_SCALE = np.array([15.0, 15.0, 8.0, 10.0, 40000.0], dtype=np.float32)

    def __init__(self, scenario="dry", timestep_hours=6.0, episode_days=1.0, noise_std=0.0, action_delay=0, seed=None, omega=None, reward_calculator=None):
        super().__init__()

        self.scenario = scenario
        self.timestep_hours = timestep_hours
        self.episode_days = episode_days
        self.noise_std = noise_std
        self.action_delay = max(0, int(action_delay))
        self.omega = omega
        self.reward_calculator = reward_calculator or RewardCalculator()

        self._data_path = str(DATA_DIR / SCENARIOS[self.scenario])
        self.bsm = BSM1OL(data_in=self._data_path)

        self._snap = _get_stable_snapshot(self._data_path)

        self.action_space = spaces.Box(low=0.0, high=360.0, shape=(3,), dtype=np.float32)
        obs_dim = 6 if self.omega is not None else 5
        self.observation_space = spaces.Box(low=0.0, high=5.0, shape=(obs_dim,), dtype=np.float32)

        self.steps_per_action = max(1, int(self.timestep_hours / 0.25))
        self._max_steps = int((self.episode_days * 24) / self.timestep_hours)
        self.reset(seed=seed)

    def _obs(self, y5):
        raw = np.array([y5[SNH], y5[SNO], y5[SO], y5[SALK], y5[Q]], dtype=np.float32)
        raw /= self._OBS_SCALE
    
        if self.omega is not None:
            raw = np.append(raw, self.omega)

        return raw

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        _restore_bsm(self.bsm, self._snap)
        
        self._cursor = self.np_random.integers(0, len(self.bsm.data_in))
        self._step = 0
        burn_klas = np.array([0.0, 0.0, 240.0, 240.0, 240.0])

        for _ in range(2 * self.steps_per_action):
            self.bsm.step(self._cursor, burn_klas)
            self._cursor = (self._cursor + 1) % len(self.bsm.data_in)
        
        if hasattr(self.reward_calculator, 'reset'):
            self.reward_calculator.reset()
        
        self._state = self._obs(self.bsm.y_out5)
        
        return self._state, {}

    def step(self, action):
        klas = np.array([0.0, 0.0, float(action[0]), float(action[1]), float(action[2])])
        total_ae = 0.0

        for _ in range(self.steps_per_action):
            self.bsm.step(self._cursor, klas)
            self._cursor = (self._cursor + 1) % len(self.bsm.data_in)
            total_ae += self.bsm.ae

        y5, y_eff = self.bsm.y_out5, self.bsm.ys_eff
        snh, sno, so = y5[SNH], y5[SNO], y5[SO]

        if self.noise_std:
            snh += self.np_random.normal(0, self.noise_std * max(abs(snh), 1e-6))
            sno += self.np_random.normal(0, self.noise_std * max(abs(sno), 1e-6))
            so  += self.np_random.normal(0, self.noise_std * max(abs(so), 1e-6))

        raw = np.array([snh, sno, so, y5[SALK], y5[Q]], dtype=np.float32)
        raw /= self._OBS_SCALE
        
        if self.omega is not None:
            raw = np.append(raw, self.omega)
        
        self._state = raw

        ntot = self.bsm.performance.advanced_quantities(y_eff, components=("totalN",))[0][0]
        avg_ae = total_ae / self.steps_per_action

        reward = self.reward_calculator.compute_composite_reward(avg_ae=avg_ae, effluent_snh=y_eff[SNH], effluent_ntot=ntot,tank_snhs=[self.bsm.y_out3[SNH], self.bsm.y_out4[SNH], y5[SNH]],klas=klas[2:],)
        
        self._step += 1

        return (self._state, float(reward), self._step >= self._max_steps, False, {"effluent_snh": y_eff[SNH], "effluent_ntot": ntot, "energy": avg_ae},)
