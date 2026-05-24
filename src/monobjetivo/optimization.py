import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt

# Representação da solução
def create_empty_solution(n_machines):
    return [[] for _ in range(n_machines)]

def clone_solution(solution):
    return [machine[:] for machine in solution]



# Avalição da Solução
def evaluate_solution(solution, pt, we, due_date):
    n_tasks = len(we)
    n_machines = len(solution)

    completion_times = np.zeros(n_tasks)
    machine_completion = np.zeros(n_machines)

    for k in range(n_machines):
        tempo_atual = 0.0
        for task in solution[k]:
            tempo_atual += pt[task, k]
            completion_times[task] = tempo_atual
        machine_completion[k] = tempo_atual

    tardiness = np.maximum(completion_times - due_date, 0.0)
    weighted_tardiness = float(np.sum(we * tardiness))
    makespan = float(np.max(machine_completion))

    return makespan, weighted_tardiness, completion_times, tardiness, machine_completion

def objective_value(solution, pt, we, due_date, objective):
    makespan, weighted_tardiness, _, _, _ = evaluate_solution(solution, pt, we, due_date)
    if objective == "f1":
        return makespan
    elif objective == "f2":
        return weighted_tardiness
    else:
        raise ValueError("objective deve ser 'f1' ou 'f2'.")
    

# Heuristica
def greedy_initial_solution(pt, we, due_date, objective, rng):
    """
    Constrói solução inicial gulosa.
    """
    n_tasks, n_machines = pt.shape
    solution = create_empty_solution(n_machines)

    tasks = list(range(n_tasks))

    if objective == "f1":
        tasks.sort(key=lambda j: -np.min(pt[j, :]))
    else:
        tasks.sort(key=lambda j: (-we[j], np.min(pt[j, :])))

    for i in range(len(tasks) - 1):
        if rng.random() < 0.15:
            j = rng.randint(i, len(tasks) - 1)
            tasks[i], tasks[j] = tasks[j], tasks[i]

    for task in tasks:
        best_sol = None
        best_val = float("inf")

        for k in range(n_machines):
            candidate = clone_solution(solution)
            candidate[k].append(task)
            val = objective_value(candidate, pt, we, due_date, objective)

            if val < best_val:
                best_val = val
                best_sol = candidate

        solution = best_sol

    return solution




# Estrutura de Vizinhança
def intra_machine_swap(solution, rng):
    sol = clone_solution(solution)
    candidate_machines = [k for k in range(len(sol)) if len(sol[k]) >= 2]

    if not candidate_machines:
        return sol

    k = rng.choice(candidate_machines)
    i, j = rng.sample(range(len(sol[k])), 2)
    sol[k][i], sol[k][j] = sol[k][j], sol[k][i]
    return sol

def inter_machine_insert(solution, rng):
    sol = clone_solution(solution)
    source_candidates = [k for k in range(len(sol)) if len(sol[k]) >= 1]

    if not source_candidates:
        return sol

    k_from = rng.choice(source_candidates)
    idx_from = rng.randrange(len(sol[k_from]))
    task = sol[k_from].pop(idx_from)

    k_to = rng.randrange(len(sol))
    idx_to = rng.randrange(len(sol[k_to]) + 1)
    sol[k_to].insert(idx_to, task)

    return sol

def inter_machine_swap(solution, rng):
    sol = clone_solution(solution)
    candidate_machines = [k for k in range(len(sol)) if len(sol[k]) >= 1]

    if len(candidate_machines) < 2:
        return sol

    k1, k2 = rng.sample(candidate_machines, 2)
    i = rng.randrange(len(sol[k1]))
    j = rng.randrange(len(sol[k2]))
    sol[k1][i], sol[k2][j] = sol[k2][j], sol[k1][i]
    return sol

NEIGHBORHOODS = [
    ("swap_intra", intra_machine_swap),
    ("insert_inter", inter_machine_insert),
    ("swap_inter", inter_machine_swap),
]




# Busca Local
def local_search(solution, pt, we, due_date, objective, rng, max_no_improve=60):
    current = clone_solution(solution)
    current_val = objective_value(current, pt, we, due_date, objective)

    no_improve = 0

    while no_improve < max_no_improve:
        improved = False
        neighborhoods = NEIGHBORHOODS[:]
        rng.shuffle(neighborhoods)

        for _, neigh in neighborhoods:
            candidate = neigh(current, rng)
            candidate_val = objective_value(candidate, pt, we, due_date, objective)

            if candidate_val < current_val:
                current = candidate
                current_val = candidate_val
                improved = True
                no_improve = 0
                break

        if not improved:
            no_improve += 1

    return current, current_val

# VNS
def run_vns(pt, we, due_date, objective="f1", max_iter=400, seed=42):
    rng = random.Random(seed)

    current = greedy_initial_solution(pt, we, due_date, objective, rng)
    current, current_val = local_search(current, pt, we, due_date, objective, rng)

    best = clone_solution(current)
    best_val = current_val

    history = [best_val]
    iter_count = 0

    while iter_count < max_iter:
        k = 0

        while k < len(NEIGHBORHOODS) and iter_count < max_iter:
            _, neigh = NEIGHBORHOODS[k]

            shaken = neigh(best, rng)

            candidate, candidate_val = local_search(
                shaken, pt, we, due_date, objective, rng
            )

            if candidate_val < best_val:
                best = candidate
                best_val = candidate_val
                k = 0
            else:
                k += 1

            history.append(best_val)
            iter_count += 1

    return best, best_val, history

# Execuções Múltiplas
def run_multiple_times(pt, we, due_date, objective, n_runs=5, max_iter=400, seed_base=42):
    results = []
    best_global_sol = None
    best_global_val = float("inf")
    best_global_history = None

    for r in range(n_runs):
        seed = seed_base + r

        best_sol, best_val, history = run_vns(
            pt, we, due_date,
            objective=objective,
            max_iter=max_iter,
            seed=seed
        )

        results.append({
            "run": r + 1,
            "seed": seed,
            "value": best_val,
            "history": history,
            "solution": best_sol
        })

        if best_val < best_global_val:
            best_global_val = best_val
            best_global_sol = best_sol
            best_global_history = history

    values = np.array([r["value"] for r in results], dtype=float)

    summary = {
        "min": float(np.min(values)),
        "std": float(np.std(values, ddof=0)),
        "max": float(np.max(values)),
        "best_solution": best_global_sol,
        "best_value": best_global_val,
        "best_history": best_global_history,
        "all_results": results
    }

    return summary
