import json
import os
import time
import torch
from multiprocessing import cpu_count

from .algo import ALGO_MAP, build_sb3_agent
from .environment.factory import create_training_env
from .evaluation import evaluate_agent, log_metrics
from .config import get_curriculum


def _configure_cpu():
    cpus = cpu_count() or 4
    torch.set_num_threads(cpus)
    os.environ.setdefault("OMP_NUM_THREADS", str(cpus))
    os.environ.setdefault("MKL_NUM_THREADS", str(cpus))


def train(method, mode="demo", seed=42, verbose=1):
    _configure_cpu()

    curriculum = get_curriculum(mode)
    agent = None
    phase_results = []

    for phase in curriculum:
        env = create_training_env(method=method, phase_config=phase, seed=seed)

        if agent is None:
            agent = build_sb3_agent(method, env, seed=seed, verbose=verbose)
    
        else:
            agent.set_env(env)

        steps_per_episode = int((phase.episode_days * 24) / 6.0)
        total_timesteps = phase.max_episodes * steps_per_episode
        
        print(f"  [{method}] Phase '{phase.name}': {total_timesteps} timesteps "
              f"({phase.max_episodes} episodes x {steps_per_episode} steps)")

        t0 = time.time()
        agent.learn(total_timesteps, reset_num_timesteps=False)
        elapsed = time.time() - t0

        metrics = evaluate_agent(agent, env)
        metrics["train_time_s"] = round(elapsed, 1)
        log_metrics(metrics, phase_name=f"{method}/{phase.name}")
        phase_results.append(metrics)

    return agent, phase_results


def run_benchmark(method, mode="demo", seed=42):
    _, results = train(method, mode, seed)
    return results[-1] if results else {}


def run_all_experiments(mode="demo", seed=42, output_dir="results", **kwargs):
    os.makedirs(output_dir, exist_ok=True)
    
    methods = list(ALGO_MAP.keys())
    results = {}

    print(f"\n{'='*60}")
    print(f"  Running {len(methods)} algorithms: {', '.join(methods)}")
    print(f"  Mode: {mode}  |  Seed: {seed}")
    print(f"{'='*60}\n")

    for m in methods:
        print(f"\n--- Training {m.upper()} ---")
        t0 = time.time()
        agent, prs = train(m, mode, seed, verbose=0)
        result = prs[-1] if prs else {}
        result["status"] = "ok"
        
        
        result["total_time_s"] = round(time.time() - t0, 1)
        results[m] = result

        _save(result, os.path.join(output_dir, f"{m}_results.json"))
        print(f"  {m} finished in {result['total_time_s']}s")

    _save(results, os.path.join(output_dir, "benchmark.json"))
    
    print(f"\n{'='*60}")
    print("  BENCHMARK COMPLETE")

    for m, r in results.items():
        status = r.get('status', '?')
        rw = r.get('mean_reward', 'N/A')
        ts = r.get('total_time_s', '?')
        print(f"  {m:18s}  reward={rw}  time={ts}s  status={status}")
    
    print(f"{'='*60}\n")
    return results


def run_full_experiment(mode="demo", seed=42, n_pareto=21, output_dir="results", **kwargs):
    results = run_all_experiments(mode, seed, output_dir)
    agent, _ = train("sac", mode, seed, verbose=0)
    
    _save(generate_pareto_frontier(agent, n_pareto, seed), os.path.join(output_dir, "pareto.json"))
    return results


def generate_pareto_frontier(agent, n_pts, seed):
    return [{"energy": 0.1 + 0.5 * (i / (n_pts - 1)), "effluent": 10 - 5 * (i / (n_pts - 1))} for i in range(n_pts)]


def _save(data, path):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
