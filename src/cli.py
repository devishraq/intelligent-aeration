import os
import argparse
import shutil
import glob

from .training import (
    train, run_benchmark, run_all_experiments,
    generate_pareto_frontier, run_full_experiment, _save,
)
from .algo import ALGO_MAP


def main(argv=None):
    parser = argparse.ArgumentParser(prog="src",
        description="Multi-objective RL for BSM1 aeration control")

    parser.add_argument("--action",
        choices=["train", "benchmark", "frontier", "full", "clean", "_worker"],
        default="benchmark")
    
    parser.add_argument("--method", default=None)
    parser.add_argument("--mode", choices=["demo", "full"], default="demo")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pareto-points", type=int, default=21)
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--list-methods", action="store_true")

    args = parser.parse_args(argv)

    if args.list_methods:
        print("Available methods:", ", ".join(ALGO_MAP.keys()))
        return

    action = args.action

    if action == "_worker":
        method = args.method or "ppo"
        agent, prs = train(method, args.mode, args.seed)
        last = prs[-1] if prs else {}
    
        os.makedirs(args.output_dir, exist_ok=True)
    
        agent.save(os.path.join(args.output_dir, "model.zip"))
    
        _save({k: v for k, v in last.items() if k != "agent"}, os.path.join(args.output_dir, "result.json"))
        return

    if action == "train":
        method = args.method or "ppo"
        train(method, args.mode, args.seed)
        print(f"Training complete: {method} ({args.mode})")

    elif action == "benchmark":
        if args.method:
            print(f"Benchmark: {run_benchmark(args.method, args.mode, args.seed)}")

        else:
            results = run_all_experiments(args.mode, args.seed, args.output_dir, n_workers=args.workers)
            print(f"Benchmarked {len(results)} methods.")

    elif action == "frontier":
        agent, _ = train("sac", args.mode, args.seed)
        frontier = generate_pareto_frontier(agent, args.pareto_points, args.seed)

        print(f"Generated {len(frontier)} frontier points.")

    elif action == "full":
        run_full_experiment(args.mode, args.seed, args.pareto_points, args.output_dir)

        print("Full experiment complete.")

    elif action == "clean":
        if os.path.exists(args.output_dir):
            shutil.rmtree(args.output_dir)
        
        for p in glob.glob("**/__pycache__", recursive=True):
            shutil.rmtree(p)
        
        print("Cleaned.")


if __name__ == "__main__":
    main()
