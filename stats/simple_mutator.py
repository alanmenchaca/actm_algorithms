import time

import matplotlib.pyplot as plt
import scienceplots

from actm_algorithms.stats.helper_func import load_seqA_and_seqB, \
    generate_populations_of_simple_mutator, append_best_fitness_from_populations_to_list
from mutation.simple_mutator import SimpleMutator

# parameters:
#   * rand_indexes_len
#   * gaps_lengths_arr
#   * num_sequences

simple_mutator: SimpleMutator = SimpleMutator()

gaps_len_range_list: list[tuple[int, int]] = [
    (1, 2), (3, 4), (5, 6), (7, 8), (9, 10)
]
rand_indexes_len_range_list: list[tuple[int, int]] = [
    (1, 2), (3, 4), (5, 6), (7, 8), (9, 10)
]

seqA, seqB = load_seqA_and_seqB()
num_sequences: int = 1500
best_fitness_params_grid: list[list[float]] = []

start_time = time.time()
for gaps_len_range in gaps_len_range_list:
    current_best_fitness_params_list: list[float] = []

    for rand_indexes_len_range in rand_indexes_len_range_list:
        params = {
            'rand_indexes_len': rand_indexes_len_range,
            'gaps_lengths_arr': gaps_len_range
        }

        seqA_population, seqB_population = \
            generate_populations_of_simple_mutator(seqA, seqB, num_sequences, params)
        append_best_fitness_from_populations_to_list(seqA_population, seqB_population,
                                                     current_best_fitness_params_list)

        print(f'gaps_len_range: {gaps_len_range}, '
              f'rand_indexes_len_range: {rand_indexes_len_range}')

    print('-' * 54)
    best_fitness_params_grid.append(current_best_fitness_params_list)

end_time = time.time()
elapsed_time_minutes = (end_time - start_time) / 60
print(f'Elapsed time: {elapsed_time_minutes} min')

# ################################################################################

plt.figure()
plt.suptitle('Mejores fitness entre las secuencias mutadas de env_HIV1H y env_HIV1S')
plt.title(f'{num_sequences * 2} secuencias por cada par de parámetros seleccionados')

plt.imshow(best_fitness_params_grid, cmap='viridis', interpolation='nearest')
plt.rcParams['font.size'] = 10

for i in range(len(gaps_len_range_list)):
    for j in range(len(rand_indexes_len_range_list)):
        plt.text(j, i, round(best_fitness_params_grid[i][j], 2),
                 ha='center', va='center', color='w')

plt.xticks(range(len(rand_indexes_len_range_list)), rand_indexes_len_range_list)
plt.yticks(range(len(gaps_len_range_list)), gaps_len_range_list)

plt.xlabel('Rango de indices aleatorios')
plt.ylabel('Rango de gaps por indice aleatorio')

cbar = plt.colorbar()
cbar.ax.tick_params(labelsize=10)
cbar.set_label('fitness', size=10)

plt.show()
