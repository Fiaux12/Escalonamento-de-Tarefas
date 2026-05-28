import random

from optimization import run_multiple_times, run_vns
from monobjetivo.visualization import *

# Configurações
FILE_PATH = "data/i5x25.xlsx"
N_RUNS = 5
MAX_ITER = 400
SEED_BASE = 42

# Exemplo de uma evaluator_config
# evaluator_config_example = {
#     "evaluator": build_weighted_tardiness_evaluator(we, pt, due_date),
#     "tasks": list(range(n_tasks)).sort(key=lambda j: (-we[j], np.min(pt[j, :]))),
#     "name": "atraso ponderado"
# }

# recebe lista de evaluator configs e retorna lista de summaries
def run_single_objective(evaluator_configs, n_machine, pt, we, due_date):
    summaries = []
    for evaluator_config in evaluator_configs:
        summary = run_multiple_times(
            evaluator_config["evaluator"], evaluator_config["tasks"], n_machine,
            n_runs=N_RUNS,
            max_iter=MAX_ITER,
            seed_base=SEED_BASE
        )

        print_summary_table(summary, evaluator_config["name"])
        plot_convergence(summary, "Curvas de convergência - " + evaluator_config["name"])
        plot_best_schedule(summary["best_solution"], "Melhor solução para " + evaluator_config["name"], pt, we, due_date)
        summaries.append(summary)
    
    return summaries

# Funcoes multiobjetivos ========================================================

# De acordo com os pesos passados como parametro, retorna a soma ponderada de f1 e f2 para a solucao corrente
def soma_ponderada(solucao_corrente, evaluator1, evaluator2, summary_f1, summary_f2, peso):
    f1, f2 = normalizar(solucao_corrente, evaluator1, evaluator2, summary_f1, summary_f2)

    return (peso * f1) + ((1-peso) * f2)


# Recebe como parametro a solucao corrente, os dois evaluators e dois summaries
# Retorna f1 e f2 normalizados
def normalizar(solucao_corrente, evaluator1, evaluator2, summary_f1, summary_f2):
    solucao_f1 = summary_f1["best_solution"]
    solucao_f2 = summary_f2["best_solution"]


    min_f1, f2_em_f1 = evaluator1(solucao_f1), evaluator2(solucao_f1)
    f1_em_f2, min_f2 = evaluator1(solucao_f2), evaluator2(solucao_f2)
    f1, f2 = evaluator1(solucao_corrente), evaluator2(solucao_corrente)

    max_f1 = f1_em_f2
    max_f2 = f2_em_f1

    f1_normalizado = (f1 - min_f1)/(max_f1 - min_f1)
    f2_normalizado = (f2 - min_f2)/(max_f2 - min_f2)

    return f1_normalizado, f2_normalizado

# Gera o summary da funcao de soma ponderada
def run_vns_soma_ponderada(evaluator1, evaluator2, summary_f1, summary_f2, peso, tasks, n_machine,
                           n_runs=5, max_iter=400, seed_base=42):
    def obj_func(sol):
        return soma_ponderada(sol, evaluator1, evaluator2, summary_f1, summary_f2, peso)
    
    return run_multiple_times(obj_func, tasks, n_machine, n_runs, max_iter, seed_base)
