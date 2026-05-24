import pandas as pd
import numpy as np
from config import *
from monobjetivo.optimization import *

'''
    min f = a*C + b*Σ(wT)
'''


def soma_ponderada(solucao_corrente, pt, we, due_date,summary_f1, summary_f2, peso):
    f1, f2 = normalizar(pt, we, due_date, summary_f1, summary_f2)

    return (peso * f1) + ((1-peso) * f2)



def normalizar(solucao_corrente, pt, we, due_date, summary_f1, summary_f2):
    solucao_f1 = summary_f1["best_solution"]
    solucao_f2 = summary_f2["best_solution"]

    min_f1, f2_em_f1, _, _, _ = evaluate_solution(solucao_f1, pt, we, due_date)
    f1_em_f2, min_f2, _, _, _ = evaluate_solution(solucao_f2, pt, we, due_date)
    f1, f2, _, _, _ = evaluate_solution(solucao_corrente, pt, we, due_date)

    max_f1 = f1_em_f2
    max_f2 = f2_em_f1

    f1_normalizado = (f1 - min_f1)/(max_f1 - min_f1)
    f2_normalizado = (f2 - min_f2)/(max_f2 - min_f2)

    return f1_normalizado, f2_normalizado