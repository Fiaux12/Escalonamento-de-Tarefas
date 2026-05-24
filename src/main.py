import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
from config import *
from optimization import *
from visualization import *

# main
def main():
    pt, we, due_date, n_tasks, n_machines = load_instance_from_excel(FILE_PATH)

    print("Instância carregada com sucesso.")
    print(f"Número de tarefas: {n_tasks}")
    print(f"Número de máquinas: {n_machines}")
    print(f"Due date: {due_date}")
    print(f"Formato da matriz PT: {pt.shape}")
    print(f"Formato do vetor WE: {we.shape}")

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


if __name__ == "__main__":
    main()