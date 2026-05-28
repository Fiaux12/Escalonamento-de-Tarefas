import pandas as pd
import numpy as np
from config import *
from optimization import *
from visualization import *
from run_functions import run_single_objective, run_vns_soma_ponderada, calcular_maximos
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


# main
def main():
    pt, we, due_date, n_tasks, n_machines = load_instance_from_excel(FILE_PATH)

    print_instance_info(n_tasks, n_machines, due_date, pt, we)
    evaluator_configs = []

    # Cria lista de tasks 
    tasks = list(range(n_tasks))
    tasks.sort(key=lambda j: -np.min(pt[j, :]))
    # Cria evaluator
    evaluator_configs.append({
        "evaluator": build_makespan_evaluator(we, pt),
        "tasks": tasks,
        "name": "f1 (makespan)"
        })

    # ====================================

    # Cria lista de tasks 
    tasks = list(range(n_tasks))
    tasks.sort(key=lambda j: (-we[j], np.min(pt[j, :])))
    # Cria evaluator
    evaluator_configs.append({
        "evaluator": build_weighted_tardiness_evaluator(we, pt, due_date),
        "tasks": tasks,
        "name": "f2 (soma ponderada dos atrasos)"
        })

    # Mono-objetivo
    summaries = run_single_objective(evaluator_configs, n_machines, pt, we, due_date)

    # ====================================

    # Multiobjetivo
    pesos = np.linspace(0, 1, 11)
    fronteira_pareto = []
    maximos = calcular_maximos(evaluator_configs, tasks, n_machines)

    for peso in pesos:
        summary = run_vns_soma_ponderada(
            maximos,
            evaluator_configs,
            summaries,
            peso,
            n_machines,
            n_runs=N_RUNS,
            max_iter=MAX_ITER,
            seed_base=SEED_BASE)
        
        sol = summary["best_solution"]
        f1 = evaluator_configs[0]["evaluator"](sol)
        f2 = evaluator_configs[1]["evaluator"](sol)
        fronteira_pareto.append((f1, f2))
        # print(f"Peso = {peso:.2f} -> f1={f1:.1f}, f2={f2:.1f}")
    
    print("Solucao pareto:", fronteira_pareto)    
    plot_pareto_frontier(fronteira_pareto)



if __name__ == "__main__":
    main()