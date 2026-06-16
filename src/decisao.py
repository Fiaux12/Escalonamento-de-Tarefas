import numpy as np
from evaluation import evaluate_makespan, evaluate_weighted_tardiness


# #  Makespan (f1)
# def get_makespan(solution, we, pt):
#     return evaluate_makespan(solution, we, pt)

# #  Atraso Ponderado (f2)
# def get_weighted_tardiness(solution, we, pt, due_date):
#     return evaluate_weighted_tardiness(solution, we, pt, due_date)






# Robustez

# Perturba pt com ruído gaussiano ±5% em N cenários
# Mede o desvio padrão de f1+f2 ao longo dos cenários
# Quanto menor mais robusta a solução
def compute_robustness(solution, we, pt, due_date, n_scenarios=30, noise_pct=0.05, seed=0):
    rng = np.random.default_rng(seed)
    values = []

    for _ in range(n_scenarios):
        noise = rng.normal(loc=1.0, scale=noise_pct, size=pt.shape)
        pt_perturbed = pt * noise
        pt_perturbed = np.maximum(pt_perturbed, 0.0) 

        f1 = evaluate_makespan(solution, we, pt_perturbed)
        f2 = evaluate_weighted_tardiness(solution, we, pt_perturbed, due_date)
        values.append(f1 + f2)

    return float(np.std(values))



# Balanceamento das Máquinas

# Desvio padrão da carga total de cada máquina
# Carga = soma dos tempos de processamento das tarefas alocadas
# Quanto menor mais balanceado

def compute_balance(solution, pt):
    machine_loads = []

    for k, tasks in enumerate(solution):
        load = sum(pt[task, k] for task in tasks)
        machine_loads.append(load)

    return float(np.std(machine_loads))



# ============================================================
# MATRIZ DE DESEMPENHO

# Recebe global_front: lista de (f1, f2, solution)
# Retorna matriz (n_alternativas x 4 critérios) e lista de soluções
def build_performance_matrix(global_front, we, pt, due_date,
                              n_scenarios=30, noise_pct=0.05):
    n = len(global_front)
    matrix = np.zeros((n, 4))

    print("\nCalculando matriz de desempenho...")

    for i, (f1, f2, sol) in enumerate(global_front):
        robustness  = compute_robustness(sol, we, pt, due_date, n_scenarios, noise_pct)
        balance     = compute_balance(sol, pt)

        matrix[i, 0] = f1
        matrix[i, 1] = f2
        matrix[i, 2] = robustness
        matrix[i, 3] = balance

        print(f"  Sol {i+1:2d}: f1={f1:.2f}  f2={f2:.2f}  "
              f"robustez={robustness:.2f}  balanceamento={balance:.2f}")

    return matrix



# NORMALIZAÇÃO 

# n_i = (x_max - x_i) / (x_max - x_min)
# Resultado: 1 = melhor, 0 = pior
def normalize_matrix(matrix):
    normalized = np.zeros_like(matrix)

    for j in range(matrix.shape[1]):
        col = matrix[:, j]
        col_min = col.min()
        col_max = col.max()

        if col_max == col_min:
            # Todos iguais: nota neutra 0.5
            normalized[:, j] = 0.5
        else:
            normalized[:, j] = (col_max - col) / (col_max - col_min)

    return normalized



def print_performance_matrix(matrix, normalized):
    headers = ["f1 (makespan)", "f2 (atraso pond.)", "Robustez", "Balanceamento"]

    print("\n===== Matriz de Desempenho (valores brutos) =====")
    print(f"{'Sol':>5} | " + " | ".join(f"{h:>18}" for h in headers))
    print("-" * 90)
    for i, row in enumerate(matrix):
        print(f"  {i+1:>3} | " + " | ".join(f"{v:>18.4f}" for v in row))

    print("\n===== Matriz Normalizada (1 = melhor, 0 = pior) =====")
    print(f"{'Sol':>5} | " + " | ".join(f"{h:>18}" for h in headers))
    print("-" * 90)
    for i, row in enumerate(normalized):
        print(f"  {i+1:>3} | " + " | ".join(f"{v:>18.4f}" for v in row))