import time

import numpy as np
import matplotlib.pyplot as plt
import scienceplots

from actm_algorithms.stats.helper_func import load_seqA_and_seqB, \
    generate_populations_of_simple_mutator, generate_populations_of_crosser_mutator, \
    append_best_fitness_from_populations_to_list, round_to_nearest_multiple
from mutation.backpack_mutator import BackpackCrosserMutator
from mutation.simple_mutator import SimpleMutator

# parameters:
#   * num_sequences_list

simple_mutator: SimpleMutator = SimpleMutator()
crosser_mutator: BackpackCrosserMutator = BackpackCrosserMutator()

seqA, seqB = load_seqA_and_seqB()
num_sequences_list: list[int] = [100, 200, 300]
sm_best_fitness_list: list[float] = []
cm_best_fitness_list: list[float] = []

start_time = time.time()
for idx, num_sequences in enumerate(num_sequences_list):
    seqA_sm_population, seqB_sm_population = \
        generate_populations_of_simple_mutator(seqA, seqB, num_sequences, {})
    append_best_fitness_from_populations_to_list(seqA_sm_population, seqB_sm_population,
                                                 sm_best_fitness_list)

    seqA_cm_population, seqB_cm_population = \
        generate_populations_of_crosser_mutator(seqA_sm_population, seqB_sm_population)
    append_best_fitness_from_populations_to_list(seqA_cm_population, seqB_cm_population,
                                                 cm_best_fitness_list)
    print(f'{(idx + 1)}/{len(num_sequences_list)} sequences done!')

end_time = time.time()
elapsed_time_minutes = (end_time - start_time) / 60
print(f'elapsed_time_minutes: {elapsed_time_minutes}')

print(f'sm_best_fitness_list: {sm_best_fitness_list}')
print(f'cm_best_fitness_list: {cm_best_fitness_list}')

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
    plt.title(f'Secuencias Totales: {sum(num_sequences_list)}', fontsize=7)

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
    plt.title(f'Secuencias Totales: {sum(num_sequences_list)}', fontsize=7)

    plt.plot(num_sequences_list, sm_best_fitness_list, linewidth=0.5, label='Mutador Simple')
    plt.plot(num_sequences_list, cm_best_fitness_list, linewidth=0.5, label='Mutador Cruzado')

    plt.scatter(num_sequences_list, sm_best_fitness_list, s=4.5)
    plt.scatter(num_sequences_list, cm_best_fitness_list, s=4.5)

    plt.xticks(fontsize=6)
    plt.yticks(y_ticks, fontsize=6)

    plt.xlabel('Número de secuencias', fontsize=7)
    plt.ylabel('fitness', fontsize=7)

    plt.legend(fontsize=5)
    plt.show()
