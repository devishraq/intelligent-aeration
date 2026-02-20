import numpy as np
import torch as th
from stable_baselines3 import TD3
from stable_baselines3.common.noise import ActionNoise, NormalActionNoise
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.buffers import ReplayBuffer

from ..config import LIMIT_SNH, LIMIT_NTOT
from ..config.hyperparameters import get_params


class CascadeActionNoise(ActionNoise):
    def __init__(self, n_actions=3, sigma=24.0, cascade_corr=-0.6, min_sigma=5.0, max_sigma=50.0):
        super().__init__()

        self.n_actions = n_actions
        self.sigma = sigma
        self.base_sigma = sigma
        self.min_sigma = min_sigma
        self.max_sigma = max_sigma
        self.cascade_corr = cascade_corr

        c = cascade_corr   
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


class LagrangianReplayBuffer(ReplayBuffer):
    def __init__(self, buffer_size, observation_space, action_space,
                 device="auto", n_envs=1, optimize_memory_usage=False,
                 handle_timeout_termination=True):
        super().__init__(
            buffer_size, observation_space, action_space,
            device=device, n_envs=n_envs,
            optimize_memory_usage=optimize_memory_usage,
            handle_timeout_termination=handle_timeout_termination,
        )

        self.constraint_snh = np.zeros((buffer_size, n_envs), dtype=np.float32)
        self.constraint_ntot = np.zeros((buffer_size, n_envs), dtype=np.float32)
        self.constraint_energy = np.zeros((buffer_size, n_envs), dtype=np.float32)

        self.lambda_snh = 1.0
        self.lambda_ntot = 2.0
        self.penalty_cap = 5.0
        self.energy_bonus_scale = 0.05
        self.ema_energy_baseline = None
        self.soft_energy_bonus = True

    def add(self, obs, next_obs, action, reward, done, infos):
        if infos is not None:
            for env_idx, info in enumerate(infos):
                if isinstance(info, dict):
                    self.constraint_snh[self.pos, env_idx] = info.get("effluent_snh", 0.0)
                    self.constraint_ntot[self.pos, env_idx] = info.get("effluent_ntot", 0.0)
                    self.constraint_energy[self.pos, env_idx] = info.get("energy", 0.0)

        super().add(obs, next_obs, action, reward, done, infos)

    def _get_samples(self, batch_inds, env=None):
        samples = super()._get_samples(batch_inds, env)

        snh_vals = self.constraint_snh[batch_inds, 0]
        ntot_vals = self.constraint_ntot[batch_inds, 0]
        energy_vals = self.constraint_energy[batch_inds, 0]

        snh_excess = np.maximum(0.0, snh_vals - LIMIT_SNH)
        ntot_excess = np.maximum(0.0, ntot_vals - LIMIT_NTOT)

        penalty = self.lambda_snh * snh_excess + self.lambda_ntot * ntot_excess

        raw_rewards = np.abs(self.rewards[batch_inds].flatten())
        cap = np.maximum(self.penalty_cap * raw_rewards, 0.3)
        penalty = np.minimum(penalty, cap)

        bonus = np.zeros_like(penalty)
        if self.ema_energy_baseline is not None and self.ema_energy_baseline > 0:
            saving = ((self.ema_energy_baseline - energy_vals) / self.ema_energy_baseline)
            
            if self.soft_energy_bonus:
                s_ok = np.maximum(0.0, 1.0 - snh_excess / 2.0)
                n_ok = np.maximum(0.0, 1.0 - ntot_excess / 2.0)
                bonus = (self.energy_bonus_scale * s_ok * n_ok * np.maximum(0.0, saving))
            else:
                compliant = (snh_excess == 0) & (ntot_excess == 0)
                bonus = np.where(compliant, self.energy_bonus_scale * np.maximum(0.0, saving),0.0)

        correction = self.to_torch((bonus - penalty).reshape(-1, 1).astype(np.float32))

        return samples._replace(rewards=samples.rewards + correction)


class ConstraintAdaptiveCallback(BaseCallback):
    def __init__(self, noise=None, ema_alpha=0.02, warmup_steps=300, lambda_lr_snh=0.01, lambda_lr_ntot=0.15, lambda_max=5.0, lambda_decay=0.995, energy_bonus_scale=0.15, penalty_cap=1.5, dual_balance=True, soft_energy_bonus=True, target_violation_rate=0.2, verbose=0):
        super().__init__(verbose)
        self.noise = noise
        self.ema_alpha = ema_alpha
        self._warmup_budget = warmup_steps
        self.lambda_snh = 1.0
        self.lambda_ntot = 2.0
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
            self.ema_energy_baseline = None

    def _sync_to_buffer(self):
        buffer = getattr(self.model, 'replay_buffer', None)
        if buffer is not None and hasattr(buffer, 'lambda_snh'):
            buffer.lambda_snh = self.lambda_snh
            buffer.lambda_ntot = self.lambda_ntot
            buffer.penalty_cap = self.penalty_cap
            buffer.energy_bonus_scale = self.energy_bonus_scale
            buffer.ema_energy_baseline = self.ema_energy_baseline
            buffer.soft_energy_bonus = self.soft_energy_bonus

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
                self.lambda_snh = min(self.lambda_max, self.lambda_snh + self.lambda_lr_snh * snh_excess)

            else:
                self.lambda_snh *= self.lambda_decay

            if ntot_excess > 0:
                self.lambda_ntot = min(self.lambda_max,
                                       self.lambda_ntot + self.lambda_lr_ntot * ntot_excess)
            else:
                self.lambda_ntot *= self.lambda_decay

            if self.dual_balance:
                
                if snh_excess == 0 and snh < 0.85 * LIMIT_SNH and ntot_excess > 0:
                    self.lambda_snh *= self.lambda_decay
                if ntot_excess == 0 and ntot < 0.95 * LIMIT_NTOT and snh_excess > 0:
                    self.lambda_ntot *= self.lambda_decay

            if self.ema_energy_baseline is None:
                self.ema_energy_baseline = energy

            else:
                self.ema_energy_baseline = (0.99 * self.ema_energy_baseline + 0.01 * energy)

            self._sync_to_buffer()

            if self.noise is not None:
            
                if self._warmup_remaining > 0:
                    self._warmup_remaining -= 1
                else:
                    snh_margin = (LIMIT_SNH - snh) / LIMIT_SNH
                    ntot_margin = (LIMIT_NTOT - ntot) / LIMIT_NTOT
                    self.noise.adapt_sigma(min(snh_margin, ntot_margin))

        if self.num_timesteps % 500 == 0 and self.total_count > 0:
            rate = (sum(self._recent_violations) / len(self._recent_violations) if self._recent_violations else 0.0)
            
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
        kwargs.setdefault("replay_buffer_class", LagrangianReplayBuffer)
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
        sigma=params.pop("noise_sigma", 24.0),
        cascade_corr=params.pop("noise_cascade_corr", -0.6),
        min_sigma=params.pop("noise_min_sigma", 5.0),
        max_sigma=params.pop("noise_max_sigma", 50.0),
    )
    params["action_noise"] = noise

    cb = ConstraintAdaptiveCallback(
        noise=noise,
        ema_alpha=params.pop("ema_alpha", 0.02),
        warmup_steps=params.pop("warmup_steps", 300),
        lambda_lr_snh=params.pop("lambda_lr_snh", 0.01),
        lambda_lr_ntot=params.pop("lambda_lr_ntot", 0.15),
        lambda_max=params.pop("lambda_max", 5.0),
        lambda_decay=params.pop("lambda_decay", 0.995),
        energy_bonus_scale=params.pop("energy_bonus_scale", 0.15),
        penalty_cap=params.pop("penalty_cap", 1.5),
        dual_balance=params.pop("dual_balance", True),
        soft_energy_bonus=params.pop("soft_energy_bonus", True),
        target_violation_rate=params.pop("target_violation_rate", 0.2),
        verbose=verbose
    )

    return CATD3("MlpPolicy", env, seed=seed, tensorboard_log=tensorboard_log, verbose=verbose, device=device, constraint_callback=cb, **params,)


def get_agent_cascade_only(env, seed=None, tensorboard_log=None, verbose=0, device="cpu", **kwargs):
    params = get_params("catd3_cascade", kwargs)

    n_actions = env.action_space.shape[0]
    noise = CascadeActionNoise(
        n_actions=n_actions,
        sigma=params.pop("noise_sigma", 24.0),
        cascade_corr=params.pop("noise_cascade_corr", -0.4),
        min_sigma=params.pop("noise_min_sigma", 5.0),
        max_sigma=params.pop("noise_max_sigma", 50.0),
    )
    params["action_noise"] = noise

    return TD3("MlpPolicy", env, seed=seed, tensorboard_log=tensorboard_log, verbose=verbose, device=device, **params,)


def get_agent_lagrangian_only(env, seed=None, tensorboard_log=None, verbose=0, device="cpu", **kwargs):
    params = get_params("catd3_lagrangian", kwargs)

    n_actions = env.action_space.shape[0]
    sigma = params.pop("noise_sigma", 24.0)
    noise = NormalActionNoise(mean=np.zeros(n_actions),sigma=sigma * np.ones(n_actions),)
    params["action_noise"] = noise

    cb = ConstraintAdaptiveCallback(
        noise=None,
        ema_alpha=params.pop("ema_alpha", 0.02),
        warmup_steps=params.pop("warmup_steps", 300),
        lambda_lr_snh=params.pop("lambda_lr_snh", 0.01),
        lambda_lr_ntot=params.pop("lambda_lr_ntot", 0.15),
        lambda_max=params.pop("lambda_max", 5.0),
        lambda_decay=params.pop("lambda_decay", 0.995),
        energy_bonus_scale=params.pop("energy_bonus_scale", 0.15),
        penalty_cap=params.pop("penalty_cap", 1.5),
        dual_balance=params.pop("dual_balance", True),
        soft_energy_bonus=params.pop("soft_energy_bonus", True),
        target_violation_rate=params.pop("target_violation_rate", 0.2),
        verbose=verbose
    )

    return CATD3("MlpPolicy", env, seed=seed, tensorboard_log=tensorboard_log, verbose=verbose, device=device, constraint_callback=cb, **params,)
