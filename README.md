# Sistema de Escalonamento de Tarefas utilizando Métodos de Otimização e Tomada de Decisão

**Trabalho desenvolvido na disciplina de Teoria da Decisão, ministrada pela professora Luiza Bernardes Real.**

## Introdução

Este projeto tem como objetivo implementar um sistema de escalonamento de tarefas em máquinas paralelas não relacionadas, buscando determinar a melhor forma de alocar tarefas entre diferentes máquinas de acordo com múltiplos critérios de otimização e tomada de decisão.

O problema abordado consiste no escalonamento de tarefas em um ambiente onde cada tarefa deve ser processada por exatamente uma máquina, enquanto cada máquina pode executar apenas uma tarefa por vez. Além disso, o tempo de processamento de uma tarefa depende da máquina escolhida para sua execução. Todas as tarefas possuem uma mesma data ideal de entrega (*due date*) e estão associadas a uma penalidade proporcional ao atraso em relação a esse prazo.

## Metodologia

Para a resolução do problema, foram implementados métodos de otimização e técnicas de tomada de decisão multicritério.

Inicialmente, foi utilizada a meta-heurística **Variable Neighborhood Search (VNS)**, uma técnica de busca em vizinhança variável combinada com métodos de busca local, com o objetivo de explorar o espaço de soluções e encontrar boas alocações de tarefas.

Em seguida, foram aplicados métodos de otimização multiobjetivo:

* **Método da Soma Ponderada**, utilizado para combinar diferentes objetivos em uma única função de avaliação por meio da atribuição de pesos;
* **Método ε-restrito**, utilizado para otimizar um objetivo principal enquanto os demais são tratados como restrições.

Por fim, foram utilizados métodos de tomada de decisão multicritério para auxiliar na seleção da melhor solução encontrada:

* **Analytic Hierarchy Process (AHP)**;
* **ELECTRE I (ELimination Et Choix Traduisant la REalité)**.

Essas técnicas permitem avaliar diferentes alternativas considerando múltiplos critérios, auxiliando na escolha de uma solução mais adequada ao contexto do problema.

## Como executar

1. Instale as dependências:
pip install -r requirements.txt

2. Execute o programa:
python main.py
