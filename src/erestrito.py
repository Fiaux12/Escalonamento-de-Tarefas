from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
from optimization import run_multiple_times

def run_vns_epsilon_restrito(epsilon, primary_evaluator, constraint_evaluator,
                              tasks, n_machines,
                              n_runs=5, max_iter=100, seed_base=42,
                              penalty_factor=1e3):
    def obj_func(sol):
        primary_val = primary_evaluator(sol)
        constraint_val = constraint_evaluator(sol)
        violation = max(0.0, constraint_val - epsilon)
        return primary_val + penalty_factor * violation

    return run_multiple_times(obj_func, tasks, n_machines, n_runs, max_iter, seed_base)


def filter_non_dominated(points):
    non_dominated = []
    for p in points:
        dominated = any(
            q[0] <= p[0] and q[1] <= p[1] and q != p
            for q in points
        )
        if not dominated:
            non_dominated.append(p)
    return sorted(non_dominated, key=lambda p: p[0])


def select_most_spread(points, n=20):
    if len(points) <= n:
        return points
    indices = np.round(np.linspace(0, len(points) - 1, n)).astype(int)
    return [points[i] for i in indices]


def run_epsilon_restrito(evaluator_configs, maximos, summaries, n_machines,
                         n_runs=5, max_iter=100, seed_base=42):

    epsilons = np.linspace(maximos[1], summaries[1]["best_value"], 11)
    print("Epsilons:", epsilons)
    all_points_by_run = defaultdict(list)

    for eps in epsilons:
        summary = run_vns_epsilon_restrito(
            epsilon=eps,
            primary_evaluator=evaluator_configs[0]["evaluator"],
            constraint_evaluator=evaluator_configs[1]["evaluator"],
            tasks=evaluator_configs[0]["tasks"][:],
            n_machines=n_machines,
            n_runs=n_runs,
            max_iter=max_iter,
            seed_base=seed_base,
        )

        for result in summary["all_results"]:
            sol = result["solution"]
            f1 = evaluator_configs[0]["evaluator"](sol)
            f2 = evaluator_configs[1]["evaluator"](sol)
            all_points_by_run[result["run"]].append((f1, f2))

    all_points = [p for pts in all_points_by_run.values() for p in pts]
    global_front = filter_non_dominated(all_points)
    global_front = select_most_spread(global_front, n=20)

    return all_points_by_run, global_front


def plot_epsilon_frontiers(all_points_by_run, global_front):
    plt.figure()

    for run_id, points in all_points_by_run.items():
        front = filter_non_dominated(points)
        f1s = [p[0] for p in front]
        f2s = [p[1] for p in front]
        plt.plot(f1s, f2s, marker='o', label=f'Run {run_id}')

    f1s = [p[0] for p in global_front]
    f2s = [p[1] for p in global_front]
    plt.plot(f1s, f2s, 'k--', linewidth=2, label='Fronteira global')

    plt.xlabel('f1 (makespan)')
    plt.ylabel('f2 (weighted tardiness)')
    plt.title('Fronteira Pareto - ε-restrito')
    plt.legend()
    plt.tight_layout()
    plt.savefig('img/pareto_epsilon.png')
    plt.close()