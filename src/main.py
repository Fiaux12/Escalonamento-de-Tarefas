import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
from config import *
from optimization import *
from monobjetivo.visualization import *
# from multiobjetivo.soma_ponderada import *

from evaluation import build_makespan_evaluator, build_weighted_tardiness_evaluator

# Leitura do arquivo
def load_instance_from_excel(file_path):
    """
    Le o arquivo Excel:
    - Coluna 0: 'Tarefa'
    - Linha 0: cabeçalho das maquinas (1,2,3,4,5)
    - Linhas seguintes: tarefas 1..25
    - Última linha: DueDate
    """

    df = pd.read_excel(file_path)

    df.columns = ["Tarefa", "M1", "M2", "M3", "M4", "M5", "Peso"]
    df = df.dropna(how="all").reset_index(drop=True)

    due_row = df[df["Tarefa"].astype(str).str.strip().str.lower() == "duedate"]
    if due_row.empty:
        raise ValueError("Não foi encontrada a linha do DueDate.")

    due_date = int(due_row.iloc[0]["M1"])
    df = df[df["Tarefa"].astype(str).str.strip().str.lower() != "duedate"].copy()
    df = df[df["Tarefa"].notna()].copy()
    df["Tarefa"] = df["Tarefa"].astype(int)
    df = df.sort_values("Tarefa").reset_index(drop=True)
    pt = df[["M1", "M2", "M3", "M4", "M5"]].astype(float).to_numpy()
    we = df["Peso"].astype(float).to_numpy()

    n_tasks = pt.shape[0]
    n_machines = pt.shape[1]

    return pt, we, due_date, n_tasks, n_machines

def print_instance_info(n_tasks, n_machines, due_date, pt, we):

    print("Instância carregada com sucesso.")
    print(f"Número de tarefas: {n_tasks}")
    print(f"Número de máquinas: {n_machines}")
    print(f"Due date: {due_date}")
    print(f"Formato PT: {pt.shape}")
    print(f"Formato WE: {we.shape}")
    print(pt)
    print(we)


def run_single_objective(n_tasks,n_machines, pt, we, due_date):
    # Cria lista de tasks 
    tasks = list(range(n_tasks))
    tasks.sort(key=lambda j: -np.min(pt[j, :]))
    # Cria evaluator
    evaluator = build_makespan_evaluator(we, pt)

    summary_f1 = run_multiple_times(
        evaluator, 
        tasks,
        n_machines,
        n_runs=N_RUNS,
        max_iter=MAX_ITER,
        seed_base=SEED_BASE
    )

    print_summary_table(summary_f1, "f1 (makespan)")
    plot_convergence(summary_f1, "Curvas de convergência - f1 (makespan)")
    plot_best_schedule(summary_f1["best_solution"], pt, we, due_date, "Melhor solução para f1 (makespan)")

    # ====================================

    # Cria lista de tasks 
    tasks = list(range(n_tasks))
    tasks.sort(key=lambda j: (-we[j], np.min(pt[j, :])))

    # # Cria evaluator
    evaluator = build_weighted_tardiness_evaluator(we, pt, due_date)
    summary_f2 = run_multiple_times(
        evaluator, 
        tasks,
        n_machines,
        n_runs=N_RUNS,
        max_iter=MAX_ITER,
        seed_base=SEED_BASE
    )

    print_summary_table(summary_f2, "f2 (soma ponderada dos atrasos)")
    plot_convergence(summary_f2, "Curvas de convergência - f2 (atraso ponderado)")
    plot_best_schedule(summary_f2["best_solution"], pt, we, due_date, "Melhor solução para f2 (atraso ponderado)")

    return summary_f1, summary_f2

# def run_multiobjective(pt, we, due_date, summary_f1, summary_f2):
#     pesos = np.linspace(0, 1, 11)
#     resultados_pareto = []

#     for peso in pesos:
#         best_sol, _, _ = run_vns_soma_ponderada(
#             pt, we, due_date, summary_f1, summary_f2, peso,
#             n_runs=N_RUNS,
#             max_iter=MAX_ITER,
#             seed_base=SEED_BASE + int(peso * 100) 
#         )
#         f1, f2, _, _, _ = evaluate_solution(best_sol, pt, we, due_date)
#         resultados_pareto.append((f1, f2))
#         print(f"\Soma ponderada. Pesos = {peso:.2f}, (f1, f2)")
#     print("Solucao pareto:", resultados_pareto)
    
#     # soma_ponderada(solucao_corrente, pt, we, due_date,summary_f1, summary_f2, peso)




# main
def main():
    pt, we, due_date, n_tasks, n_machines = load_instance_from_excel(FILE_PATH)

    print_instance_info(n_tasks, n_machines, due_date, pt, we)

    get_rng = lambda : random.Random(42)

    # Comparando funcao desacoplada: makespan
    # Greedy antigo
    # res = greedy_initial_solution(pt, we, due_date, "f1", get_rng())
    # print(res)

    # # Greedy novo
    # # Cria lista de tasks 
    # tasks = list(range(n_tasks))
    # tasks.sort(key=lambda j: -np.min(pt[j, :]))
    # # Cria evaluator
    # evaluator = build_makespan_evaluator(we, pt)
    # res = gis(evaluator, tasks, n_machines, get_rng())
    # print(res)

    # # ====================================

    # # Comparando funcao desacoplada: weighted_tardiness
    # # Greedy antigo
    # # res = greedy_initial_solution(pt, we, due_date, "f2", get_rng())
    # # print(res)

    # # Greedy novo
    # # Cria lista de tasks 
    # tasks = list(range(n_tasks))
    # tasks.sort(key=lambda j: (-we[j], np.min(pt[j, :])))
    # # Cria evaluator
    # evaluator = build_weighted_tardiness_evaluator(we, pt, due_date)
    # res = gis(evaluator, tasks, n_machines, get_rng())
    # print(res)

    # # Mono-objetivo
    solucao_f1, solucao_f2 = run_single_objective(n_tasks, n_machines, pt, we, due_date)

    # # Multiobjetivo
    # run_multiobjective(pt, we, due_date, solucao_f1, solucao_f2)

if __name__ == "__main__":
    main()