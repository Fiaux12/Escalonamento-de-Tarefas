def create_empty_solution(n_machines):
    return [[] for _ in range(n_machines)]

def clone_solution(solution):
    return [machine[:] for machine in solution]

# Heuristica
# Tendo como base um evaluator (funcao que recebe solucao e retorna um valor)
# Configuracao inicial de tasks
# Quantidade de maquinas
# rng, gerador de numeros aleatorios
#
# Retorna uma solucao inicial gulosa
def greedy_initial_solution(evaluator, tasks, n_machines, rng):
    """
    Constrói solução inicial gulosa.

    Recebe o evaluator, responsavel por receber uma solucao e avaliar o valor de retorno da solucao
    tasks, lista de tasks já ordenadas inicialmente
    n_machines, quantidade de maquinas
    rng, gerador de numeros aleatorios
    """
    solution = create_empty_solution(n_machines)

    for i in range(len(tasks) - 1):
        if rng.random() < 0.15:
            j = rng.randint(i, len(tasks) - 1)
            tasks[i], tasks[j] = tasks[j], tasks[i]



    for task in tasks:
        best_sol = None
        best_val = float("inf")

        for k in range(n_machines):
            candidate = clone_solution(solution)
            candidate[k].append(task)
            val = evaluator(candidate)

            if val < best_val:
                best_val = val
                best_sol = candidate

        solution = best_sol

    return solution