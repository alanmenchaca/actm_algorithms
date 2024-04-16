import time

import matplotlib.pyplot as plt

from mutation.simple_mutator import SimpleMutator
from utils.metrics import SeqsSimilarity
from utils.seq import Sequence
from utils.seqs_manager import SeqLoader

import scienceplots

# parameters:
#   * rand_indexes_len
#   * gaps_lengths_arr
#   * num_sequences

seq1: Sequence = SeqLoader.load("../src/sequences/env_HIV1H.txt")
seq2: Sequence = SeqLoader.load("../src/sequences/env_HIV1S.txt")

num_seqs: int = 10_000
best_similarities_grid: list[list[int]] = []

gaps_len_range_list: list[tuple[int, int]] = [
    (1, 2), (3, 4), (5, 6), (7, 8), (9, 10)
]
rand_indexes_len_range_list: list[tuple[int, int]] = [
    (1, 2), (3, 4), (5, 6), (7, 8), (9, 10)
]

start_time = time.time()
for gaps_len_range in gaps_len_range_list:
    best_similarities: list[int] = []
    for rand_indexes_len_range in rand_indexes_len_range_list:
        SimpleMutator.set_params(rand_indexes_len_range, gaps_len_range)
        seq2_sm_seqs: list[Sequence] = SimpleMutator.generate_mutated_seqs(seq2, num_seqs)

        SeqsSimilarity.compute(seq1, seq2_sm_seqs)
        best_similarities.append(seq2_sm_seqs[0].similarity)

        print(f'gaps_len_range: {gaps_len_range}, '
              f'rand_indexes_len_range: {rand_indexes_len_range} '
              f'best_similarity: {seq2_sm_seqs[0].similarity}')

    print(best_similarities)
    print('-' * 54)
    best_similarities_grid.append(best_similarities)

end_time = time.time()
elapsed_time_minutes = (end_time - start_time) / 60
print(f'Elapsed time: {elapsed_time_minutes} min')

# ################################################################################


plt.figure(figsize=(7, 5))
font_size = 10

with plt.style.context(['science', 'ieee']):
    plt.rcParams['font.size'] = font_size

    plt.title(f'{num_seqs} secuencias mutadas por cada par de parámetros seleccionados', fontsize=12)
    plt.suptitle('Secuencias Mutadas del Virus del VIH del Simio con Mejor Grado de Similitud', fontsize=12)
    plt.imshow(best_similarities_grid, cmap='viridis', interpolation='nearest')

    for i in range(len(gaps_len_range_list)):
        for j in range(len(rand_indexes_len_range_list)):
            plt.text(j, i, str(round(best_similarities_grid[i][j], 2)),
                     ha='center', va='center', color='w')

    plt.xticks(range(len(rand_indexes_len_range_list)), rand_indexes_len_range_list, fontsize=font_size)
    plt.yticks(range(len(gaps_len_range_list)), gaps_len_range_list, fontsize=font_size)

    plt.xlabel('intervalo de indices aleatorios', fontsize=font_size)
    plt.ylabel('intervalo de gaps por indice aleatorio', fontsize=font_size)

    cbar = plt.colorbar()
    cbar.ax.tick_params(labelsize=font_size)  # colorbar tick label size
    cbar.set_label('similaridad', size=font_size)  # bar label

    plt.show()
