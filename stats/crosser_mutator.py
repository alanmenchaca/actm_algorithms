import time

import matplotlib.pyplot as plt
import numpy as np

from mutation.crosser_mutator import CrosserMutator as CMutator
from mutation.simple_mutator import SimpleMutator
from utils.seqs_manager import SeqLoader
from utils.metrics import SeqsSimilarity
from utils.seq import Sequence

import scienceplots

# parameters:
#   * num_sequences_list

seq1: Sequence = SeqLoader.load("../src/sequences/env_HIV1S.txt")
seq2: Sequence = SeqLoader.load("../src/sequences/env_HIV1H.txt")

num_seqs_list: list[int] = [100, 150, 200, 250, 300]
sm_best_fitness_list: list[float] = []
cm_best_fitness_list: list[float] = []

start_time = time.time()
for idx, num_seqs in enumerate(num_seqs_list):
    seq2_sm_seqs: list[Sequence] = SimpleMutator.generate_mutated_seqs(seq2, num_seqs)
    seq2_cm_seqs: list[Sequence] = CMutator.generate_mutated_seqs(seq2_sm_seqs)

    SeqsSimilarity.compute(seq1, seq2_sm_seqs)
    SeqsSimilarity.compute(seq1, seq2_cm_seqs)

    sm_best_fitness_list.append(seq2_sm_seqs[0].similarity)
    cm_best_fitness_list.append(seq2_cm_seqs[0].similarity)

    print(f'{(idx + 1)}/{len(num_seqs_list)} sequences done!')

end_time = time.time()
elapsed_time_minutes = (end_time - start_time) / 60
print(f'elapsed_time_minutes: {elapsed_time_minutes}')
print(f'sm_best_fitness_list: {sm_best_fitness_list}')
print(f'cm_best_fitness_list: {cm_best_fitness_list}')


def round_to_nearest_multiple(value: int, base: int = 100) -> int:
    remainder: int = value % base
    if remainder <= (base / 2):
        return value - remainder
    else:
        return value + (base - remainder)


all_best_fitness = np.concatenate((sm_best_fitness_list, cm_best_fitness_list))
min_nearest_multiple = round_to_nearest_multiple(min(all_best_fitness))
max_nearest_multiple = round_to_nearest_multiple(max(all_best_fitness))
y_ticks = np.arange(min_nearest_multiple, (max_nearest_multiple + 100), 100)

# ################################################################################
# first graph: boxplot

with plt.style.context(['science', 'ieee', 'grid']):
    plt.figure()
    plt.rcParams['font.size'] = 7

    plt.suptitle('Rendimiento de Mutador Simple y Mutador Cruzado', fontsize=7)
    plt.title(f'Secuencias Totales: {sum(num_seqs_list)}', fontsize=7)

    plt.boxplot([sm_best_fitness_list, cm_best_fitness_list],
                labels=['Mutador Simple', 'Mutador Cruzado'],
                showfliers=False)

    plt.xticks(fontsize=6)
    plt.yticks(y_ticks, fontsize=6)

    plt.ylabel('fitness', fontsize=7)
    plt.show()

# ################################################################################
# second graph: scatter

with plt.style.context(['science', 'ieee', 'grid']):
    plt.figure()
    plt.suptitle('Rendimiento  de Mutador Simple y Mutador Cruzado', fontsize=7)
    plt.title(f'Secuencias Totales: {sum(num_seqs_list)}', fontsize=7)

    plt.plot(num_seqs_list, sm_best_fitness_list, linewidth=0.5, label='Mutador Simple')
    plt.plot(num_seqs_list, cm_best_fitness_list, linewidth=0.5, label='Mutador Cruzado')

    plt.scatter(num_seqs_list, sm_best_fitness_list, s=4.5)
    plt.scatter(num_seqs_list, cm_best_fitness_list, s=4.5)

    plt.xticks(fontsize=6)
    plt.yticks(y_ticks, fontsize=6)

    plt.xlabel('Número de secuencias', fontsize=7)
    plt.ylabel('fitness', fontsize=7)

    plt.legend(fontsize=5)
    plt.show()
