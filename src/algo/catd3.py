import numpy as np
from stable_baselines3 import TD3
from stable_baselines3.common.noise import ActionNoise
from stable_baselines3.common.callbacks import BaseCallback

from ..config import LIMIT_SNH, LIMIT_NTOT
from ..config.hyperparameters import get_params


class CascadeActionNoise(ActionNoise):

    def __init__(self, n_actions=3, sigma=30.0, cascade_corr=0.3, min_sigma=3.0, max_sigma=60.0):
        super().__init__()

        self.n_actions = n_actions
        self.sigma = sigma
        self.base_sigma = sigma
        self.min_sigma = min_sigma
        self.max_sigma = max_sigma
        self.cascade_corr = cascade_corr

        c = cascade_corr
        
        corr = np.array([
            [1.0, c, c**2],
            [c, 1.0, c],
            [c**2, c, 1.0],
        ], dtype=np.float64)
        
        self._L = np.linalg.cholesky(corr)
        self._step = 0

    def __call__(self):
        z = np.random.randn(self.n_actions)
        correlated = (self._L @ z) * self.sigma
        self._step += 1
        return correlated.astype(np.float32)

    def reset(self):
        pass

    def adapt_sigma(self, constraint_margin):
        if constraint_margin > 0.5:
            self.sigma = min(self.max_sigma, self.sigma * 1.002)
        elif constraint_margin < 0.0:
            self.sigma = max(self.min_sigma, self.sigma * 0.998)
        else:
            self.sigma += 0.01 * (self.base_sigma - self.sigma)


class ConstraintAdaptiveCallback(BaseCallback):

    def __init__(self, noise=None, ema_alpha=0.01, warmup_steps=500, verbose=0):
        super().__init__(verbose)
        self.noise = noise
        self.ema_alpha = ema_alpha
        self._warmup_budget = warmup_steps
        self.ema_snh = 0.0
        self.ema_ntot = 0.0
        self.ema_energy = 0.0
        self.violation_count = 0
        self.total_count = 0
        self._phase_count = 0
        self._warmup_remaining = 0

    def _on_training_start(self):
        self._phase_count += 1
        if self._phase_count > 1 and self.noise is not None:
            self._warmup_remaining = self._warmup_budget
            self.noise.sigma = min(self.noise.max_sigma, self.noise.base_sigma * 2.0)

    def _on_step(self):
        infos = self.locals.get("infos", [])

        for info in infos:
        
            snh = info.get("effluent_snh")
        
            if snh is None:
                continue
        
            ntot = info["effluent_ntot"]
        
            energy = info["energy"]

            a = self.ema_alpha
        
            self.ema_snh = (1 - a) * self.ema_snh + a * snh
            self.ema_ntot = (1 - a) * self.ema_ntot + a * ntot
            self.ema_energy = (1 - a) * self.ema_energy + a * energy

            self.total_count += 1
            
            if snh > LIMIT_SNH or ntot > LIMIT_NTOT:
                self.violation_count += 1

            if self.noise is not None:
                if self._warmup_remaining > 0:
                    self._warmup_remaining -= 1
                else:
                    snh_margin = (LIMIT_SNH - snh) / LIMIT_SNH
                    ntot_margin = (LIMIT_NTOT - ntot) / LIMIT_NTOT
                    self.noise.adapt_sigma(min(snh_margin, ntot_margin))

        if self.num_timesteps % 500 == 0 and self.total_count > 0:
            rate = self.violation_count / self.total_count
            self.logger.record("catd3/ema_snh", round(self.ema_snh, 4))
            self.logger.record("catd3/ema_ntot", round(self.ema_ntot, 4))
            self.logger.record("catd3/ema_energy", round(self.ema_energy, 2))
            self.logger.record("catd3/violation_rate", round(rate, 4))
            self.logger.record("catd3/phase", self._phase_count)
            if self.noise:
                self.logger.record("catd3/sigma", round(self.noise.sigma, 4))
        return True


class CATD3(TD3):

    def __init__(self, *args, constraint_callback=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._constraint_callback = constraint_callback

    def learn(self, total_timesteps, callback=None, **kwargs):
        if self._constraint_callback is not None:
            if callback is None:
                callback = self._constraint_callback
            elif isinstance(callback, list):
                callback = callback + [self._constraint_callback]
            else:
                callback = [callback, self._constraint_callback]
        return super().learn(total_timesteps, callback=callback, **kwargs)


def get_agent(env, seed=None, tensorboard_log=None, verbose=0, device="cpu", **kwargs):
    params = get_params("catd3", kwargs)

    n_actions = env.action_space.shape[0]
    noise = CascadeActionNoise(
        n_actions=n_actions,
        sigma=30.0,
        cascade_corr=0.3,
        min_sigma=3.0,
        max_sigma=60.0,
    )
    params["action_noise"] = noise

    cb = ConstraintAdaptiveCallback(noise=noise, verbose=verbose)

    return CATD3(
        "MlpPolicy",
        env,
        seed=seed,
        tensorboard_log=tensorboard_log,
        verbose=verbose,
        device=device,
        constraint_callback=cb,
        **params,
    )
