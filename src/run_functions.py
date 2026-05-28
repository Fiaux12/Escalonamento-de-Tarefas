from optimization import run_multiple_times
from visualization import *

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
def soma_ponderada(peso, f1, f2):
    return (peso * f1) + ((1-peso) * f2)

# Recebe como parametro a solucao corrente, os dois evaluators e dois summaries
# Retorna f1 e f2 normalizados
def normalizar(solucao_corrente, evaluators, summaries, maximos):
    min_f1 = summaries[0]["best_value"]
    min_f2 = summaries[1]["best_value"]

    f1 = evaluators[0]["evaluator"](solucao_corrente)
    f2 = evaluators[1]["evaluator"](solucao_corrente)

    f1_normalizado = (f1 - min_f1)/(maximos[0] - min_f1)
    f2_normalizado = (f2 - min_f2)/(maximos[1] - min_f2)

    return f1_normalizado, f2_normalizado

# Calcula os maximos de f1 e f2 (ponto nadir)
def calcular_maximos(evaluators, tasks, n_machines,
                   n_runs=5, max_iter=100, seed_base=42):
    
    maximos = []
    
    # Maximiza f1 minimizando -f1
    summary_max_f1 = run_multiple_times(
        lambda sol: -evaluators[0]["evaluator"](sol), tasks, n_machines, n_runs, max_iter, seed_base
    )
    # O valor real é o negativo do mínimo encontrado
    maximos.append(-summary_max_f1["min"])

    # Maximiza f2 minimizando -f2
    summary_max_f2 = run_multiple_times(
        lambda sol: -evaluators[1]["evaluator"](sol), tasks, n_machines, n_runs, max_iter, seed_base
    )
    maximos.append(-summary_max_f2["min"])

    return maximos

# Gera o summary da funcao de soma ponderada
def run_vns_soma_ponderada(maximos, evaluators, summaries, peso, n_machines,
                           n_runs=5, max_iter=100, seed_base=42):
    
    if peso >= 0.5:
        tasks = evaluators[0]["tasks"][:]   # prioriza f1
    else:
        tasks = evaluators[1]["tasks"][:]   # prioriza f2


    def obj_func(sol):
        f1, f2 = normalizar(sol, evaluators, summaries, maximos)
        return soma_ponderada(peso, f1, f2)
    
    return run_multiple_times(obj_func, tasks, n_machines, n_runs, max_iter, seed_base)
