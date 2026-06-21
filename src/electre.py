import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def validar_entradas(matrix_normalizada, pesos):
    matrix_normalizada = np.asarray(matrix_normalizada, dtype=float)
    pesos = np.asarray(pesos, dtype=float)

    if matrix_normalizada.ndim != 2:
        raise ValueError("A matriz normalizada deve ter duas dimensões.")

    if len(matrix_normalizada) < 2:
        raise ValueError("O ELECTRE I precisa de pelo menos duas alternativas.")

    if matrix_normalizada.shape[1] != len(pesos):
        raise ValueError("A quantidade de pesos deve ser igual à quantidade de critérios.")

    if np.any(pesos < 0):
        raise ValueError("Os pesos não podem ser negativos.")

    if pesos.sum() == 0:
        raise ValueError("A soma dos pesos não pode ser zero.")

    pesos = pesos / pesos.sum()

    return matrix_normalizada, pesos


def calcular_concordancia(matrix_normalizada, pesos):
    matrix_normalizada, pesos = validar_entradas(matrix_normalizada, pesos)
    n_alternativas = matrix_normalizada.shape[0]

    concordancia = np.zeros((n_alternativas, n_alternativas))

    for a in range(n_alternativas):
        for b in range(n_alternativas):
            if a == b:
                concordancia[a, b] = 1.0
            else:
                criterios_favoraveis = matrix_normalizada[a, :] >= matrix_normalizada[b, :]
                concordancia[a, b] = pesos[criterios_favoraveis].sum()

    return concordancia


def calcular_discordancia(matrix_normalizada):
    matrix_normalizada = np.asarray(matrix_normalizada, dtype=float)
    n_alternativas = matrix_normalizada.shape[0]

    discordancia = np.zeros((n_alternativas, n_alternativas))

    for a in range(n_alternativas):
        for b in range(n_alternativas):
            if a == b:
                discordancia[a, b] = 0.0
            else:
                perdas = matrix_normalizada[b, :] - matrix_normalizada[a, :]
                discordancia[a, b] = max(0.0, np.max(perdas))

    return discordancia


def calcular_sobreclassificacao(
    concordancia,
    discordancia,
    limiar_concordancia=0.60,
    limiar_discordancia=0.40
):
    concordancia = np.asarray(concordancia, dtype=float)
    discordancia = np.asarray(discordancia, dtype=float)

    sobreclassificacao = (
        (concordancia >= limiar_concordancia)
        & (discordancia <= limiar_discordancia)
    )

    np.fill_diagonal(sobreclassificacao, False)

    return sobreclassificacao


def obter_nucleo(sobreclassificacao):
    """
    Núcleo = alternativas que não são sobreclassificadas por nenhuma outra.

    Se S[a,b] = True, então a sobreclassifica b.
    Logo, b é superada se existe algum a tal que S[a,b] = True.
    """
    sobreclassificacao = np.asarray(sobreclassificacao, dtype=bool)

    superada = sobreclassificacao.any(axis=0)

    nucleo = []
    for i, foi_superada in enumerate(superada):
        if not foi_superada:
            nucleo.append(i)

    return nucleo


def escolher_solucao_final(nucleo, matrix_normalizada, pesos, matrix_bruta=None):
    matrix_normalizada, pesos = validar_entradas(matrix_normalizada, pesos)

    scores = matrix_normalizada @ pesos

    if len(nucleo) == 0:
        candidatos = list(range(matrix_normalizada.shape[0]))
    else:
        candidatos = nucleo

    # Critério adicional: menor atraso ponderado
    # Coluna 1 da matriz bruta = atraso ponderado
    if matrix_bruta is not None:
        matrix_bruta = np.asarray(matrix_bruta, dtype=float)
        melhor_indice = min(candidatos, key=lambda i: matrix_bruta[i, 1])
    else:
        melhor_indice = max(candidatos, key=lambda i: scores[i])

    return melhor_indice, scores


def executar_electre_i(
    matrix_normalizada,
    pesos,
    matrix_bruta=None,
    limiar_concordancia=0.60,
    limiar_discordancia=0.40
):
    matrix_normalizada, pesos = validar_entradas(matrix_normalizada, pesos)

    concordancia = calcular_concordancia(matrix_normalizada, pesos)
    discordancia = calcular_discordancia(matrix_normalizada)

    sobreclassificacao = calcular_sobreclassificacao(
        concordancia,
        discordancia,
        limiar_concordancia,
        limiar_discordancia
    )

    nucleo = obter_nucleo(sobreclassificacao)

    melhor_indice, scores = escolher_solucao_final(
        nucleo,
        matrix_normalizada,
        pesos,
        matrix_bruta
    )

    return {
        "pesos": pesos,
        "concordancia": concordancia,
        "discordancia": discordancia,
        "sobreclassificacao": sobreclassificacao,
        "nucleo": nucleo,
        "scores": scores,
        "melhor_indice": melhor_indice,
        "melhor_solucao": melhor_indice + 1,
        "limiar_concordancia": limiar_concordancia,
        "limiar_discordancia": limiar_discordancia,
    }


def imprimir_resultado_electre(resultado):
    nomes_criterios = [
        "Makespan",
        "Atraso ponderado",
        "Robustez",
        "Balanceamento"
    ]

    print("\n===== RESULTADO ELECTRE I =====")

    print("\nPesos utilizados:")
    for nome, peso in zip(nomes_criterios, resultado["pesos"]):
        print(f"{nome:20s}: {peso:.4f}")

    print(f"\nLimiar de concordância: {resultado['limiar_concordancia']:.2f}")
    print(f"Limiar de discordância: {resultado['limiar_discordancia']:.2f}")

    print("\nMatriz de concordância:")
    print(np.round(resultado["concordancia"], 4))

    print("\nMatriz de discordância:")
    print(np.round(resultado["discordancia"], 4))

    print("\nMatriz de sobreclassificação:")
    print(resultado["sobreclassificacao"].astype(int))

    nucleo_humano = [i + 1 for i in resultado["nucleo"]]
    print(f"\nNúcleo do ELECTRE I: {nucleo_humano}")

    print("\nScores usados apenas para desempate:")
    for i, score in enumerate(resultado["scores"]):
        print(f"Sol {i+1:2d}: {score:.4f}")

    print(f"\nSolução escolhida pelo ELECTRE I: Sol {resultado['melhor_solucao']}")


def plot_fronteira_electre(global_front, resultado, output_path="src/img/fronteira_electre.png"):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    melhor_indice = resultado["melhor_indice"]

    f1s = [p[0] for p in global_front]
    f2s = [p[1] for p in global_front]

    plt.figure(figsize=(8, 5))
    plt.plot(f1s, f2s, marker="o", linestyle="--", label="Soluções avaliadas")
    plt.scatter(
        f1s[melhor_indice],
        f2s[melhor_indice],
        s=180,
        marker="*",
        label="Escolha ELECTRE I"
    )

    for i, (f1, f2, _) in enumerate(global_front):
        plt.text(f1, f2, f" Sol {i+1}", fontsize=9)

    plt.xlabel("f1 (Makespan)")
    plt.ylabel("f2 (Atraso ponderado)")
    plt.title("Fronteira avaliada com solução escolhida pelo ELECTRE I")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_gantt_electre(global_front, resultado, pt, due_date, output_path="src/img/gantt_electre.png"):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    melhor_indice = resultado["melhor_indice"]
    solution = global_front[melhor_indice][2]

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
    plt.title(f"Solução escolhida pelo ELECTRE I - Sol {melhor_indice + 1}")
    plt.yticks(range(1, len(solution) + 1), [f"M{k}" for k in range(1, len(solution) + 1)])
    plt.legend()
    plt.grid(True, axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()                     