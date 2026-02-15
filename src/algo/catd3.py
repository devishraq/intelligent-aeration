import numpy as np
from stable_baselines3 import TD3
from stable_baselines3.common.noise import ActionNoise
from stable_baselines3.common.callbacks import BaseCallback

from ..config import LIMIT_SNH, LIMIT_NTOT
from ..config.hyperparameters import get_params


class CascadeActionNoise(ActionNoise):
    def __init__(self, n_actions=3, sigma=24.0, cascade_corr=-0.4,
                 min_sigma=5.0, max_sigma=50.0):
        super().__init__()

        self.n_actions = n_actions
        self.sigma = sigma
        self.base_sigma = sigma
        self.min_sigma = min_sigma
        self.max_sigma = max_sigma
        self.cascade_corr = cascade_corr

        c = cascade_corr  # e.g. -0.4
        corr = np.array([
            [1.0,     c,    c * 0.5],
            [c,       1.0,  c      ],
            [c * 0.5, c,    1.0    ],
        ], dtype=np.float64)

        eigvals = np.linalg.eigvalsh(corr)
        if eigvals.min() < 1e-6:
            corr += (1e-6 - eigvals.min()) * np.eye(self.n_actions)

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
            self.sigma = max(self.min_sigma, self.sigma * 0.995)
        elif constraint_margin < 0.0:
            self.sigma = min(self.max_sigma, self.sigma * 1.01)
        else:
            self.sigma += 0.05 * (self.base_sigma - self.sigma)


class ConstraintAdaptiveCallback(BaseCallback):
    def __init__(self, noise=None, ema_alpha=0.02, warmup_steps=300,
                 lambda_lr_snh=0.008, lambda_lr_ntot=0.03,
                 lambda_max=1.5, lambda_decay=0.993,
                 energy_bonus_scale=0.15, penalty_cap=0.4,
                 dual_balance=False, soft_energy_bonus=False,
                 target_violation_rate=0.25, verbose=0):
        super().__init__(verbose)
        self.noise = noise
        self.ema_alpha = ema_alpha
        self._warmup_budget = warmup_steps
        self.lambda_snh = 0.1     
        self.lambda_ntot = 0.2    
        self.lambda_lr_snh = lambda_lr_snh
        self.lambda_lr_ntot = lambda_lr_ntot
        self.lambda_max = lambda_max
        self.lambda_decay = lambda_decay
        self.energy_bonus_scale = energy_bonus_scale
        self.penalty_cap = penalty_cap
        self.dual_balance = dual_balance
        self.soft_energy_bonus = soft_energy_bonus
        self.target_violation_rate = target_violation_rate

        self.ema_snh = 0.0
        self.ema_ntot = 0.0
        self.ema_energy = 0.0
        self.ema_energy_baseline = None  
        self.violation_count = 0
        self.total_count = 0
        self._phase_count = 0
        self._warmup_remaining = 0
        self._recent_violations = []  

    def _on_training_start(self):
        self._phase_count += 1
        if self._phase_count > 1:

            self._warmup_remaining = self._warmup_budget
            if self.noise is not None:
                self.noise.sigma = min(self.noise.max_sigma,
                                       self.noise.base_sigma * 1.5)
            self.lambda_snh = 0.1
            self.lambda_ntot = 0.2
            self.ema_energy_baseline = None

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
            violated = snh > LIMIT_SNH or ntot > LIMIT_NTOT
            if violated:
                self.violation_count += 1
            self._recent_violations.append(1.0 if violated else 0.0)
            if len(self._recent_violations) > 200:
                self._recent_violations.pop(0)

            snh_excess = max(0.0, snh - LIMIT_SNH)
            ntot_excess = max(0.0, ntot - LIMIT_NTOT)

            if snh_excess > 0:
                self.lambda_snh = min(self.lambda_max,
                                      self.lambda_snh + self.lambda_lr_snh * snh_excess)
            else:
                self.lambda_snh *= self.lambda_decay

            if ntot_excess > 0:
                self.lambda_ntot = min(self.lambda_max,
                                       self.lambda_ntot + self.lambda_lr_ntot * ntot_excess)
            else:
                self.lambda_ntot *= self.lambda_decay


            if self.dual_balance:
                if snh_excess == 0 and snh < 0.85 * LIMIT_SNH and ntot_excess > 0:
                    self.lambda_snh *= self.lambda_decay   # extra decay
                if ntot_excess == 0 and ntot < 0.95 * LIMIT_NTOT and snh_excess > 0:
                    self.lambda_ntot *= self.lambda_decay   # extra decay

            if "rewards" in self.locals:
                lagrangian_penalty = (
                    self.lambda_snh * snh_excess +
                    self.lambda_ntot * ntot_excess
                )

                base_reward = abs(float(self.locals["rewards"][0]))
                max_penalty = max(self.penalty_cap * base_reward, 0.3)
                lagrangian_penalty = min(lagrangian_penalty, max_penalty)

                energy_bonus = 0.0
                if self.ema_energy_baseline is None:
                    self.ema_energy_baseline = energy
                else:
                    self.ema_energy_baseline = (
                        0.99 * self.ema_energy_baseline + 0.01 * energy
                    )
                energy_saving = (self.ema_energy_baseline - energy) / max(self.ema_energy_baseline, 1.0)

                if self.soft_energy_bonus:
                    snh_ok = max(0.0, 1.0 - snh_excess / 2.0)
                    ntot_ok = max(0.0, 1.0 - ntot_excess / 2.0)
                    compliance_frac = snh_ok * ntot_ok
                    energy_bonus = self.energy_bonus_scale * compliance_frac * max(0.0, energy_saving)
                else:
                    if snh_excess == 0 and ntot_excess == 0:
                        energy_bonus = self.energy_bonus_scale * max(0.0, energy_saving)

                self.locals["rewards"][0] += energy_bonus - lagrangian_penalty

            if self.noise is not None:
                if self._warmup_remaining > 0:
                    self._warmup_remaining -= 1
                else:
                    snh_margin = (LIMIT_SNH - snh) / LIMIT_SNH
                    ntot_margin = (LIMIT_NTOT - ntot) / LIMIT_NTOT
                    self.noise.adapt_sigma(min(snh_margin, ntot_margin))

        if self.num_timesteps % 500 == 0 and self.total_count > 0:
            rate = (sum(self._recent_violations) / len(self._recent_violations)
                    if self._recent_violations else 0.0)
            self.logger.record("catd3/ema_snh", round(self.ema_snh, 4))
            self.logger.record("catd3/ema_ntot", round(self.ema_ntot, 4))
            self.logger.record("catd3/ema_energy", round(self.ema_energy, 2))
            self.logger.record("catd3/violation_rate", round(rate, 4))
            self.logger.record("catd3/lambda_snh", round(self.lambda_snh, 4))
            self.logger.record("catd3/lambda_ntot", round(self.lambda_ntot, 4))
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
        sigma=24.0,
        cascade_corr=-0.4,
        min_sigma=5.0,
        max_sigma=50.0,
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
