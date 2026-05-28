import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
from config import *
from optimization import *
from evaluation import *

# Plots dos gráficos
def plot_convergence(results_summary, title):
    plt.figure(figsize=(8, 5))
    for item in results_summary["all_results"]:
        plt.plot(item["history"], label=f"Execução {item['run']}")
    plt.xlabel("Iterações")
    plt.ylabel("Valor da função objetivo")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("img/convergencia.png")
    plt.close()

def plot_best_schedule(solution, pt, we, due_date, title="Melhor solução"):
    makespan = evaluate_makespan(solution, we, pt)
    weighted_tardiness = evaluate_weighted_tardiness(solution, we, pt, due_date)

    plt.figure(figsize=(10, 5))

    for k, machine_tasks in enumerate(solution):
        tempo_atual = 0.0
        for task in machine_tasks:
            duration = pt[task, k]
            plt.barh(
                y=k + 1,
                width=duration,
                left=tempo_atual,
                edgecolor="black"
            )
            plt.text(
                tempo_atual + duration / 2,
                k + 1,
                str(task + 1),
                ha="center",
                va="center",
                fontsize=8
            )
            tempo_atual += duration

    plt.axvline(due_date, linestyle="--", label=f"Due date = {due_date}")
    plt.xlabel("Tempo")
    plt.ylabel("Máquina")
    plt.title(title)
    plt.yticks(range(1, len(solution) + 1), [f"M{k}" for k in range(1, len(solution) + 1)])
    plt.legend()
    plt.grid(True, axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig("img/schedule.png")
    plt.close()

    print("Resumo da melhor solução:")
    print(f"Makespan = {makespan:.2f}")
    print(f"Atraso ponderado = {weighted_tardiness:.2f}")
    # print(f"Tempos finais das máquinas = {machine_completion}")
    print(solution_to_string(solution))


# Resultados
def print_summary_table(summary, objective_name):
    print(f"\n===== Resultados para {objective_name} =====")
    for item in summary["all_results"]:
        print(f"Execução {item['run']}: valor final = {item['value']:.4f}")
    print(f"\nmin = {summary['min']:.4f}")
    print(f"std = {summary['std']:.4f}")
    print(f"max = {summary['max']:.4f}")


def solution_to_string(solution):
    lines = []
    for k, tasks in enumerate(solution):
        tarefas_humanas = [t + 1 for t in tasks]
        lines.append(f"Máquina {k+1}: {tarefas_humanas}")
    return "\n".join(lines)