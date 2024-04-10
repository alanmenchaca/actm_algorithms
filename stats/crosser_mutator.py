import time

import matplotlib.pyplot as plt
import numpy as np

from mutation.crosser_mutator import CrosserMutator as CMutator
from mutation.simple_mutator import SimpleMutator
from utils.metrics import SeqsSimilarity
from utils.seq import Sequence
from utils.seqs_manager import SeqLoader

import scienceplots

# parameters:
#   * num_sequences_list

seq1: Sequence = SeqLoader.load("../src/sequences/env_HIV1S.txt")
seq2: Sequence = SeqLoader.load("../src/sequences/env_HIV1H.txt")

num_seqs_list: list[int] = [100, 150, 200, 250, 300]
sm_best_similarities: list[float] = []
cm_best_similarities: list[float] = []

start_time = time.time()
for idx, num_seqs in enumerate(num_seqs_list):
    seq2_sm_seqs: list[Sequence] = SimpleMutator.generate_mutated_seqs(seq2, num_seqs)
    seq2_cm_seqs: list[Sequence] = CMutator.generate_mutated_seqs(seq2_sm_seqs)

    SeqsSimilarity.compute(seq1, seq2_sm_seqs)
    SeqsSimilarity.compute(seq1, seq2_cm_seqs)

    sm_best_similarities.append(seq2_sm_seqs[0].similarity)
    cm_best_similarities.append(seq2_cm_seqs[0].similarity)

    print(f'{(idx + 1)}/{len(num_seqs_list)} sequences done!')

end_time = time.time()
elapsed_time_minutes = (end_time - start_time) / 60
print(f'elapsed_time_minutes: {elapsed_time_minutes}')
print(f'sm_best_similarities: {sm_best_similarities}')
print(f'cm_best_similarities: {cm_best_similarities}')


def round_to_nearest_multiple(value: int, base: int = 100) -> int:
    remainder: int = value % base
    if remainder <= (base / 2):
        return value - remainder
    else:
        return value + (base - remainder)


all_best_similarities = np.concatenate((sm_best_similarities, cm_best_similarities))
min_nearest_multiple = round_to_nearest_multiple(min(all_best_similarities))
max_nearest_multiple = round_to_nearest_multiple(max(all_best_similarities))
y_ticks = np.arange(min_nearest_multiple - 150, (max_nearest_multiple + 150), 50)

# ################################################################################
# first graph: boxplot
plt.figure(figsize=(8, 6))
with plt.style.context(['science', 'ieee', 'grid']):
    plt.rcParams['font.size'] = 7

    plt.suptitle('Rendimiento de Mutador Simple y Mutador Cruzado', fontsize=10)
    plt.title(f'Secuencias Totales: {sum(num_seqs_list)}', fontsize=10)

    plt.boxplot([sm_best_similarities, cm_best_similarities],
                labels=['Mutador Simple', 'Mutador Cruzado'],
                showfliers=False)

    plt.xticks(fontsize=8)
    plt.yticks(y_ticks, fontsize=8)

    plt.ylabel('similaridad', fontsize=10)
    plt.show()

# ################################################################################
# second graph: scatter
plt.figure(figsize=(7, 5))
with plt.style.context(['science', 'ieee', 'grid']):
    plt.suptitle('Rendimiento  de Mutador Simple y Mutador Cruzado', fontsize=10)
    plt.title(f'Secuencias Totales: {sum(num_seqs_list)}', fontsize=10)

    plt.plot(num_seqs_list, sm_best_similarities, linewidth=0.5, label='Mutador Simple')
    plt.plot(num_seqs_list, cm_best_similarities, linewidth=0.5, label='Mutador Cruzado')

    plt.scatter(num_seqs_list, sm_best_similarities, s=4.5)
    plt.scatter(num_seqs_list, cm_best_similarities, s=4.5)

    plt.xticks(fontsize=8)
    plt.yticks(y_ticks, fontsize=8)

    plt.xlabel('número de secuencias', fontsize=10)
    plt.ylabel('similaridad', fontsize=10)

    plt.legend(fontsize=10)
    plt.show()
