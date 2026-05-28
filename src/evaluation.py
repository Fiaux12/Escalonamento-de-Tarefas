import numpy as np

# constroi a funcao weighted de forma que ela receba apenas o argumento solution
def build_weighted_tardiness_evaluator(we, pt, due_date):
    return lambda solution: evaluate_weighted_tardiness(solution, we, pt, due_date)

# Avalição da Weighted Tardiness (F2)
def evaluate_weighted_tardiness(solution, we, pt, due_date):
    n_tasks = len(we)
    n_machines = len(solution)

    completion_times = np.zeros(n_tasks)

    for k in range(n_machines):
        tempo_atual = 0.0
        for task in solution[k]:
            tempo_atual += pt[task, k]
            completion_times[task] = tempo_atual

    tardiness = np.maximum(completion_times - due_date, 0.0)
    weighted_tardiness = float(np.sum(we * tardiness))

    return weighted_tardiness

# Constroi funcao makespan de forma que ela receba apenas o argumento solution
def build_makespan_evaluator(we, pt):
    return lambda solution: evaluate_makespan(solution, we, pt)

# Avalição de Makespan (F1)
def evaluate_makespan(solution, we, pt):
    n_tasks = len(we)
    n_machines = len(solution)

    completion_times = np.zeros(n_tasks)
    machine_completion = np.zeros(n_machines)

    for k in range(n_machines):
        tempo_atual = 0.0
        for task in solution[k]:
            tempo_atual += pt[task, k]
            completion_times[task] = tempo_atual
        machine_completion[k] = tempo_atual

    makespan = float(np.max(machine_completion))

    return makespan
