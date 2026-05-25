from monobjetivo.optimization import *

'''
    min f = a*C + b*Σ(wT)
'''

def soma_ponderada(solucao_corrente, pt, we, due_date,summary_f1, summary_f2, peso):
    f1, f2 = normalizar(solucao_corrente, pt, we, due_date, summary_f1, summary_f2)

    return (peso * f1) + ((1-peso) * f2)



def normalizar(solucao_corrente, pt, we, due_date, summary_f1, summary_f2):
    solucao_f1 = summary_f1["best_solution"]
    solucao_f2 = summary_f2["best_solution"]

    min_f1, f2_em_f1, _, _, _ = evaluate_solution(solucao_f1, pt, we, due_date)
    f1_em_f2, min_f2, _, _, _ = evaluate_solution(solucao_f2, pt, we, due_date)
    f1, f2, _, _, _ = evaluate_solution(solucao_corrente, pt, we, due_date)

    max_f1 = f1_em_f2
    max_f2 = f2_em_f1

    f1_normalizado = (f1 - min_f1)/(max_f1 - min_f1)
    f2_normalizado = (f2 - min_f2)/(max_f2 - min_f2)

    return f1_normalizado, f2_normalizado

def run_vns_soma_ponderada(pt, we, due_date, summary_f1, summary_f2, peso, 
                           n_runs=5, max_iter=400, seed_base=42):
    def obj_func(sol):
        return soma_ponderada(sol, pt, we, due_date, summary_f1, summary_f2, peso)

    best_global_sol = None
    best_global_val = float("inf")
    best_global_history = None

    for r in range(n_runs):
        seed = seed_base + r
        sol, val, history = run_vns(pt, we, due_date, obj_func, 
                                    objective="f2", max_iter=max_iter, seed=seed)

        if val < best_global_val:
            best_global_val = val
            best_global_sol = sol
            best_global_history = history

    return best_global_sol, best_global_val, best_global_history                                                     