import numpy as np
from ..config import LIMIT_SNH, LIMIT_NTOT


class RewardCalculator:
    def __init__(self, energy_weight=0.5, snh_violation_weight=0.2,
                 ntot_violation_weight=0.1, energy_scale=1000.0):
        self.energy_weight = energy_weight
        self.snh_violation_weight = snh_violation_weight
        self.ntot_violation_weight = ntot_violation_weight
        self.energy_scale = energy_scale

    def compute_composite_reward(self, avg_ae, effluent_snh, effluent_ntot, **kwargs):
        energy_cost = (avg_ae / self.energy_scale) * self.energy_weight
        snh_cost = max(0.0, effluent_snh - LIMIT_SNH) * self.snh_violation_weight
        ntot_cost = max(0.0, effluent_ntot - LIMIT_NTOT) * self.ntot_violation_weight
        return -(energy_cost + snh_cost + ntot_cost)


class MultiObjectiveRewardCalculator(RewardCalculator):
    def __init__(self, omega=0.5, **kwargs):
        super().__init__(**kwargs)
        self.omega = omega

    def compute_composite_reward(self, avg_ae, effluent_snh, effluent_ntot, **kwargs):
        energy_cost = (avg_ae / self.energy_scale) * self.energy_weight
        violation_cost = (
            max(0.0, effluent_snh - LIMIT_SNH) * self.snh_violation_weight +
            max(0.0, effluent_ntot - LIMIT_NTOT) * self.ntot_violation_weight
        )
        return -((1 - self.omega) * energy_cost + self.omega * violation_cost)


class ZAMORRewardCalculator:

    def __init__(self, energy_scale=1000.0, barrier_weight=0.5, cascade_weight=0.15,
                 smoothness_weight=0.05, ntot_penalty_weight=1.5,
                 buffer_snh=2.5, buffer_ntot=14.0):
        self.energy_scale = energy_scale
        self.barrier_weight = barrier_weight
        self.cascade_weight = cascade_weight
        self.smoothness_weight = smoothness_weight
        self.ntot_penalty_weight = ntot_penalty_weight
        self.buffer_snh = buffer_snh
        self.buffer_ntot = buffer_ntot
        self._prev_klas = None

    def reset(self):
        self._prev_klas = None

    def compute_composite_reward(self, avg_ae, effluent_snh, effluent_ntot,
                                 tank_snhs=None, klas=None, **_extra):
        energy_ratio = avg_ae / self.energy_scale
        energy_cost = np.log1p(energy_ratio * 5.0) / 3.0

        snh_cost = self._zone_penalty(effluent_snh, self.buffer_snh, LIMIT_SNH)
        ntot_cost = self._zone_penalty(effluent_ntot, self.buffer_ntot, LIMIT_NTOT) * self.ntot_penalty_weight

        barrier = (
            self._log_barrier(effluent_snh, LIMIT_SNH)
            + self._log_barrier(effluent_ntot, LIMIT_NTOT)
        )

        cascade_bonus = 0.0
        if tank_snhs is not None:
            cascade_bonus = self._cascade_efficiency(tank_snhs)

        smoothness_cost = 0.0
        if klas is not None:
            if self._prev_klas is not None:
                delta = np.abs(np.asarray(klas) - self._prev_klas)
                smoothness_cost = float(np.sum(delta)) / 360.0
            self._prev_klas = np.asarray(klas, dtype=np.float64).copy()

        total_cost = (
            energy_cost
            + snh_cost
            + ntot_cost
            + self.barrier_weight * barrier
            + self.smoothness_weight * smoothness_cost
            - self.cascade_weight * cascade_bonus
        )
        return -total_cost

    @staticmethod
    def _zone_penalty(value, buf, limit):
        if value <= buf:
            return 0.01 * value / limit
        if value <= limit:
            t = (value - buf) / (limit - buf)
            return 0.5 * t * t
        excess = value - limit
        return 0.5 + 2.0 * excess + 0.5 * excess * excess

    @staticmethod
    def _log_barrier(value, limit, scale=0.1):
        margin = max(limit - value, 1e-4) / limit
        if margin > 0.5:
            return 0.0
        return -scale * np.log(margin + 1e-8)

    @staticmethod
    def _cascade_efficiency(tank_snhs):
        snhs = list(tank_snhs)
        if len(snhs) < 2:
            return 0.0
        bonus = 0.0
        for i in range(len(snhs) - 1):
            if snhs[i] > snhs[i + 1]:
                reduction = (snhs[i] - snhs[i + 1]) / max(snhs[i], 1e-6)
                bonus += reduction
        return bonus / (len(snhs) - 1)
