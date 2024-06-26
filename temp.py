import matplotlib.pyplot as plt
import numpy as np

from utils.seq import Sequence
from utils.seqs_manager import SeqLoader

import scienceplots

seq1: Sequence = SeqLoader.load("src/sequences/env_HIV1H.txt")
seq2: Sequence = SeqLoader.load("src/sequences/env_HIV1S.txt")

num_sequences_list: list[int] = [200, 300, 400, 500, 600, 700, 800]
total_sequences: int = 10_179

sm_best_similarities: list[float] = [939, 1614, 1532, 999, 1530, 1573, 1487]
cm_best_similarities: list[float] = [1267, 1561, 1569, 1415, 1412, 1633, 1487]
cr_best_similarities: list[float] = [1264, 1686, 1421, 1282, 1297, 1549, 1575]
sa_best_similarities: list[float] = [811, 805, 768, 721, 1245, 787, 889]

print(f'sm_best_similarities: {sm_best_similarities}')
print(f'cm_best_similarities: {cm_best_similarities}')
print(f'cr_best_similarities: {cr_best_similarities}')
print(f'sa_best_similarities: {sa_best_similarities}')


def round_to_nearest_multiple(value: int, base: int = 100) -> int:
    remainder: int = value % base
    if remainder <= (base / 2):
        return value - remainder
    else:
        return value + (base - remainder)


all_best_similarities = np.concatenate((sm_best_similarities, cm_best_similarities,
                                        cr_best_similarities, sa_best_similarities))
min_nearest_multiple = round_to_nearest_multiple(min(all_best_similarities))
max_nearest_multiple = round_to_nearest_multiple(max(all_best_similarities))
y_ticks = np.arange(min_nearest_multiple, (max_nearest_multiple + 100), 100)

# ################################################################################

plt.figure(figsize=(7, 5))
font_size = 10

with plt.style.context(['science', 'ieee', 'grid']):
    plt.suptitle('Similaridad en cada Metaheurística de Optimización', fontsize=12)
    plt.title(f'Secuencias Totales: {total_sequences}', fontsize=12)

    plt.boxplot([sm_best_similarities, cm_best_similarities,
                 cr_best_similarities, sa_best_similarities],
                labels=['MSS', 'MSS + MCS', 'MSS + MRQ', 'MSS + RS'])

    plt.xticks(fontsize=font_size)
    plt.yticks(y_ticks, fontsize=font_size)

    plt.xlabel('Algoritmo de Optimización', fontsize=font_size)
    plt.ylabel('similaridad', fontsize=font_size)

    plt.show()

# ################################################################################

plt.figure(figsize=(7, 5))
font_size = 10

with plt.style.context(['science', 'ieee', 'grid']):
    plt.suptitle('Grado de Similitud en cada Metaheurística de Optimización', fontsize=12)
    plt.title(f'Secuencias Totales: {total_sequences}', fontsize=12)

    plt.plot(num_sequences_list, sm_best_similarities, linestyle='dashed')
    plt.plot(num_sequences_list, cm_best_similarities, linestyle='dashdot')
    plt.plot(num_sequences_list, cr_best_similarities, linestyle='dashed')
    plt.plot(num_sequences_list, sa_best_similarities, linestyle='dashdot')

    plt.xticks(num_sequences_list, fontsize=font_size)
    plt.yticks(y_ticks, fontsize=font_size)

    plt.xlabel('Número de Secuencias', fontsize=font_size)
    plt.ylabel('similaridad', fontsize=font_size)

    plt.legend(['MSS', 'MSS + MCS', 'MSS + MRQ', 'RS'], fontsize=7)
    plt.show()
