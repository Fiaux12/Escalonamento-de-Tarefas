import numpy as np
from evaluation import evaluate_makespan, evaluate_weighted_tardiness


# #  Makespan (f1)
# def get_makespan(solution, we, pt):
#     return evaluate_makespan(solution, we, pt)

# #  Atraso Ponderado (f2)
# def get_weighted_tardiness(solution, we, pt, due_date):
#     return evaluate_weighted_tardiness(solution, we, pt, due_date)


# tabela de índices de consistência (RI) para matrizes de comparação par a par
RI_TABLE = {1: 0.00, 2: 0.00, 3: 0.58, 4: 0.90, 5: 1.12,
            6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45, 10: 1.49}

CRITERIA_NAMES = ["f1 (makespan)", "f2 (atraso pond.)", "Robustez", "Balanceamento"]

# Matriz de comparação par a par para os critérios
AHP_COMPARISON_MATRIX = np.array([
    [1,    1,    3,    4  ],
    [1,    1,    3,    4  ],
    [1/3,  1/3,  1,    2  ],
    [1/4,  1/4,  1/2,  1  ],
])
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

## AHP Analytic Hierarchy Process
def compute_ahp_weights(comparison_matrix):
    n = comparison_matrix.shape[0]

    col_sum = comparison_matrix.sum(axis=0)
    normalized = comparison_matrix / col_sum
    weights = normalized.mean(axis=1)

    lambda_max = np.mean((comparison_matrix @ weights) / weights)
    CI = (lambda_max - n) / (n - 1) if n > 2 else 0.0
    RI = RI_TABLE.get(n, 1.49)
    CR = CI / RI if RI != 0 else 0.0

    return weights, lambda_max, CI, RI, CR

def print_ahp_results(criteria_names, comparison_matrix, weights, lambda_max, CI, RI, CR):
    print("\n===== AHP: Matriz de Comparação Par-a-Par =====")
    print(f"{'':>20} | " + " | ".join(f"{c:>16}" for c in criteria_names))
    for i, row in enumerate(comparison_matrix):
        print(f"{criteria_names[i]:>20} | " + " | ".join(f"{v:>16.3f}" for v in row))

    print("\n===== AHP: Pesos dos Critérios =====")
    for c, w in zip(criteria_names, weights):
        print(f"  {c:<20} peso = {w:.4f}")

    print("\n===== AHP: Verificação de Consistência =====")
    print(f"  lambda_max = {lambda_max:.4f}")
    print(f"  CI = {CI:.4f}")
    print(f"  RI (n={len(criteria_names)}) = {RI:.2f}")
    status = "CONSISTENTE" if CR < 0.10 else "INCONSISTENTE - revisar julgamentos!"
    print(f"  CR = {CR:.4f}  -> {status}")

def compute_ahp_scores(matrix, pesos_ahp):
    normalized = normalize_matrix(matrix)
    scores = normalized @ pesos_ahp
    return normalized, scores
 

def print_ahp_ranking(matrix, pesos_ahp, criteria_names):
    normalized, scores = compute_ahp_scores(matrix, pesos_ahp)
 
    ranking = np.argsort(-scores)  # ordem decrescente de score
 
    print("\n===== Ranking final via AHP (soma ponderada) =====")
    print(f"{'Pos':>4} | {'Sol':>4} | {'Score':>8} | Detalhe por criterio")
    print("-" * 80)
 
    for pos, i in enumerate(ranking, start=1):
        detalhe = "  ".join(
            f"{criteria_names[j]}={normalized[i, j]:.3f}*{pesos_ahp[j]:.3f}"
            for j in range(len(criteria_names))
        )
        print(f"{pos:>4} | {i + 1:>4} | {scores[i]:>8.4f} | {detalhe}")
 
    melhor = ranking[0] + 1
    print(f"\n>> Melhor solucao segundo o AHP: Sol {melhor} (score = {scores[ranking[0]]:.4f})")
 
    return scores, ranking
