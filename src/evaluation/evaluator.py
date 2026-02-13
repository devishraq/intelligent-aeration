import numpy as np
from collections import defaultdict
from stable_baselines3.common.evaluation import evaluate_policy

from ..environment.bsm1 import BSM1Env
from ..environment.rewards import ZAMORRewardCalculator
from ..config import LIMITS, LIMIT_SNH, LIMIT_NTOT


def evaluate_agent(agent, env, n_eval_episodes=3, deterministic=True):
    mean_reward, std_reward = evaluate_policy(agent, env, n_eval_episodes=n_eval_episodes, deterministic=deterministic,)
    return {"mean_reward": float(mean_reward), "std_reward": float(std_reward)}


def evaluate_physical(agent, scenario, n_episodes=5, seed=42):
    rc = ZAMORRewardCalculator()
    env = BSM1Env(scenario=scenario, seed=seed, reward_calculator=rc)
    all_eps = []

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        done = False
        d = defaultdict(list)

        while not done:
            action, _ = agent.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            bsm = env.bsm
            eff = bsm.ys_eff
            aq = bsm.performance.advanced_quantities(
                eff, components=("kjeldahlN", "totalN", "COD", "BOD5e", "X_TSS")
            )[0]
            kjN, totN, cod, bod5e, xtss = aq
            ae, pe, me = bsm.ae * 24, bsm.pe * 24, bsm.me * 24

            d["snh"].append(float(eff[9]))
            d["sno"].append(float(eff[8]))
            d["ntot"].append(float(totN))
            d["cod"].append(float(cod))
            d["bod5"].append(float(bod5e))
            d["tss"].append(float(eff[13]))
            d["ae"].append(float(ae))
            d["pe"].append(float(pe))
            d["me"].append(float(me))
            d["eqi"].append(float(bsm.performance.eqi(eff)[0]))
            d["oci"].append(ae + pe + me)
            d["reward"].append(float(reward))
            d["kla3"].append(float(action[0]))
            d["kla4"].append(float(action[1]))
            d["kla5"].append(float(action[2]))
            d["snh_t3"].append(float(bsm.y_out3[9]))
            d["snh_t4"].append(float(bsm.y_out4[9]))
            d["snh_t5"].append(float(bsm.y_out5[9]))
        
        all_eps.append(d)

    avg, ts = {}, {}
    
    for key in all_eps[0]:
        per_ep = [np.mean(ep[key]) for ep in all_eps]
        avg[key] = float(np.mean(per_ep))
        avg[f"{key}_std"] = float(np.std(per_ep))
        ts[key] = all_eps[-1][key]

    total_steps = sum(len(ep["snh"]) for ep in all_eps)
    
    for vname, key, lim in [
        ("violation_snh", "snh", LIMIT_SNH),
        ("violation_ntot", "ntot", LIMIT_NTOT),
        ("violation_tss", "tss", LIMITS["TSS"]),
        ("violation_cod", "cod", LIMITS["COD"]),
        ("violation_bod5", "bod5", LIMITS["BOD5"]),
    ]:
        avg[vname] = sum(sum(1 for v in ep[key] if v > lim) for ep in all_eps) / total_steps

    return avg, ts


def log_metrics(metrics, phase_name=""):
    prefix = f"[{phase_name}] " if phase_name else ""
    
    print(f"{prefix}reward={metrics.get('mean_reward', 0):.2f} "
          f"(+/-{metrics.get('std_reward', 0):.2f})")
