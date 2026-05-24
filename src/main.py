import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
from config import *
from monobjetivo.optimization import *
from monobjetivo.visualization import *

def run_single_objective(pt, we, due_date):
    # f1 = makespan
    summary_f1 = run_multiple_times(
        pt, we, due_date,
        objective="f1",
        n_runs=N_RUNS,
        max_iter=MAX_ITER,
        seed_base=SEED_BASE
    )

    print_summary_table(summary_f1, "f1 (makespan)")
    plot_convergence(summary_f1, "Curvas de convergência - f1 (makespan)")
    plot_best_schedule(summary_f1["best_solution"], pt, we, due_date, "Melhor solução para f1 (makespan)")

    # f2 = soma ponderada dos atrasos
    summary_f2 = run_multiple_times(
        pt, we, due_date,
        objective="f2",
        n_runs=N_RUNS,
        max_iter=MAX_ITER,
        seed_base=SEED_BASE + 1000
    )

    print_summary_table(summary_f2, "f2 (soma ponderada dos atrasos)")
    plot_convergence(summary_f2, "Curvas de convergência - f2 (atraso ponderado)")
    plot_best_schedule(summary_f2["best_solution"], pt, we, due_date, "Melhor solução para f2 (atraso ponderado)")



# def run_multiobjective(pt, we, due_date):



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

    print_instance_info(
        n_tasks,
        n_machines,
        due_date,
        pt,
        we
    )

    # Mono-objetivo
    run_single_objective(pt, we, due_date)

    # Multiobjetivo
    # run_multiobjective(pt, we, due_date)

    

if __name__ == "__main__":
    main()