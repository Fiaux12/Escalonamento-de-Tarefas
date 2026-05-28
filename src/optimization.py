import numpy as np
import random
from heuristic import greedy_initial_solution as gis

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

def intra_machine_2opt(solution, rng):
    sol = clone_solution(solution)
    candidate_machines = [k for k in range(len(sol)) if len(sol[k]) >= 3]

    if not candidate_machines:
        return sol

    k = rng.choice(candidate_machines)
    i, j = sorted(rng.sample(range(len(sol[k])), 2))
    sol[k][i:j+1] = sol[k][i:j+1][::-1]
    return sol

def or_opt(solution, rng):
    sol = clone_solution(solution)
    candidate_machines = [k for k in range(len(sol)) if len(sol[k]) >= 2]

    if not candidate_machines:
        return sol

    k_from = rng.choice(candidate_machines)
    idx = rng.randrange(len(sol[k_from]) - 1)
    block = sol[k_from][idx:idx+2]
    sol[k_from] = sol[k_from][:idx] + sol[k_from][idx+2:]

    k_to = rng.choice([k for k in range(len(sol)) if k != k_from])
    pos = rng.randrange(len(sol[k_to]) + 1)
    sol[k_to] = sol[k_to][:pos] + block + sol[k_to][pos:]

    return sol

# TODO: (melhoria da otimização) add mais vizinhanca
NEIGHBORHOODS = [
    ("swap_intra", intra_machine_swap),
    ("insert_inter", inter_machine_insert),
    ("swap_inter", inter_machine_swap),
    ("2opt_intra", intra_machine_2opt), 
    ("or_opt", or_opt),
]

# Copia de funcao clone usada em heuristic
def clone_solution(solution):
    return [machine[:] for machine in solution]

# Busca Local
#
# Recebe uma solucao, o evaluator
# Realiza a operacao de busca na vizinhaca diversas vezes, ate que nao consiga mais melhorar
#
# Retorna a melhor solucao encontrada, junto com o seu valor de acordo com evaluator
def local_search(solution, evaluator, rng, max_no_improve=60):
    current = clone_solution(solution)
    current_val = evaluator(current)

    no_improve = 0

    while no_improve < max_no_improve:
        improved = False
        neighborhoods = NEIGHBORHOODS[:]
        rng.shuffle(neighborhoods)

        # TODO:Em vez de aceitar o primeiro que melhora, avalia N candidatos e pega o melhor
        # best improvement
        best_candidate = None
        best_candidate_val = current_val

        for _, neigh in neighborhoods:
            candidates = [neigh(current, rng) for _ in range(5)]
            local_best = min(candidates, key=evaluator)
            local_val = evaluator(local_best)


            if local_val < best_candidate_val:
                best_candidate_val = local_val
                best_candidate = local_best

        if best_candidate is not None:
            current = best_candidate
            current_val = best_candidate_val
            no_improve = 0
        else:
            no_improve += 1

    return current, current_val

# VNS
#
#
def run_vns(evaluator, tasks, n_machine, rng, max_iter=400):
    current = gis(evaluator, tasks, n_machine, rng)
    current, current_val = local_search(current, evaluator, rng, max_no_improve=60)

    best = clone_solution(current)
    best_val = current_val

    history = [best_val]
    iter_count = 0

    while iter_count < max_iter:
        k = 0

        while k < len(NEIGHBORHOODS) and iter_count < max_iter:
            # _, neigh = NEIGHBORHOODS[k]

            # TODO: (melhoria da otimização) Aplica mais movimentos aleatórios 
            # Shake: k controla intensidade, mais tentativas sem melhora = shake maior
            n_shakes = rng.randint(k + 2, k + 5)
            shaken = clone_solution(best)
            for _ in range(n_shakes):
                _, neigh = rng.choice(NEIGHBORHOODS)
                shaken = neigh(shaken, rng)

            candidate, candidate_val = local_search(
                shaken, evaluator, rng, max_no_improve=60
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
def run_multiple_times(evaluator, tasks, n_machine, n_runs=5, max_iter=400, seed_base=42):
    results = []
    best_global_sol = None
    best_global_val = float("inf")
    best_global_history = None

    for r in range(n_runs):
        seed = seed_base + r
        # TODO: posso variar a seed para ter resultados diferentes a cada execucao
        rng = random.Random(seed)

        best_sol, best_val, history = run_vns(evaluator, tasks, n_machine, rng, max_iter)

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