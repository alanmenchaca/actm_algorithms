import time

import numpy as np
import matplotlib.pyplot as plt
import scienceplots

from actm_algorithms.stats.helper_func import load_seqA_and_seqB, \
    generate_populations_of_simple_mutator, generate_populations_of_crosser_mutator, \
    generate_populations_of_cr_mutator, run_simulated_annealing, \
    append_best_fitness_from_populations_to_list, round_to_nearest_multiple

seqA, seqB = load_seqA_and_seqB()
# num_sequences_list: list[int] = [10, 20, 30, 40, 50]
num_sequences_list: list[int] = [200, 300, 400, 500, 600, 700, 800]
num_collisions: int = 100
total_sequences: int = 0

sm_best_fitness_list: list[float] = []  # simple mutator
cm_best_fitness_list: list[float] = []  # crosser mutator
cr_best_fitness_list: list[float] = []  # chemical reactions mutator
sa_best_fitness_list: list[float] = []  # simulated annealing

start_time = time.time()
for idx, num_sequences in enumerate(num_sequences_list):
    seqA_sm_population, seqB_sm_population = \
        generate_populations_of_simple_mutator(seqA, seqB, num_sequences, {})
    append_best_fitness_from_populations_to_list(seqA_sm_population, seqB_sm_population,
                                                 sm_best_fitness_list)

    seqA_cm_population, seqB_cm_population = \
        generate_populations_of_crosser_mutator(seqA_sm_population.copy(), seqB_sm_population.copy())
    append_best_fitness_from_populations_to_list(seqA_cm_population, seqB_cm_population,
                                                 cm_best_fitness_list)
    seqA_cr_population, seqB_cr_population = \
        generate_populations_of_cr_mutator(seqA_sm_population.copy(), seqB_sm_population.copy(), num_collisions)
    append_best_fitness_from_populations_to_list(seqA_cr_population, seqB_cr_population,
                                                 cr_best_fitness_list)

    simple_mutator_seq_len: int = len(seqA_sm_population) * 2
    crosser_mutator_seq_len: int = len(seqA_cm_population) * 2
    chemical_reactions_mutator_seq_len: int = len(seqA_cr_population) + len(seqB_cr_population)

    total_sequences += simple_mutator_seq_len + crosser_mutator_seq_len + chemical_reactions_mutator_seq_len
    print(f'total_sequences: {total_sequences}')

    best_sa_sequence = run_simulated_annealing(seqA_sm_population)
    sa_best_fitness_list.append(best_sa_sequence.fitness)
    # sa_best_fitness_list.append(np.random.randint(0, 1000))

    print(f'num. sequences completed: {(idx + 1)}/{len(num_sequences_list)}\n')

end_time = time.time()
elapsed_time_minutes = (end_time - start_time) / 60
print(f'\nelapsed_time_minutes: {elapsed_time_minutes}')

print(f'sm_best_fitness_list: {sm_best_fitness_list}')
print(f'cm_best_fitness_list: {cm_best_fitness_list}')
print(f'cr_best_fitness_list: {cr_best_fitness_list}')
print(f'sa_best_fitness_list: {sa_best_fitness_list}')

all_best_fitness = np.concatenate((sm_best_fitness_list, cm_best_fitness_list,
                                   cr_best_fitness_list, sa_best_fitness_list))
min_nearest_multiple = round_to_nearest_multiple(min(all_best_fitness))
max_nearest_multiple = round_to_nearest_multiple(max(all_best_fitness))
y_ticks = np.arange(min_nearest_multiple, (max_nearest_multiple + 100), 100)

with plt.style.context(['science', 'ieee', 'grid']):
    plt.figure()
    plt.suptitle('Mejores fitness de Secuencias en cada Mutador', fontsize=7)
    plt.title(f'Secuencias Totales: {total_sequences}', fontsize=7)

    plt.boxplot([sm_best_fitness_list, cm_best_fitness_list,
                 cr_best_fitness_list, sa_best_fitness_list],
                labels=['MSS', 'MSS + MCS', 'MSS + MRQ', 'MSS + RS'])

    plt.xticks(fontsize=6)
    plt.yticks(y_ticks, fontsize=6)

    plt.xlabel('Mutadores de secuencias', fontsize=7)
    plt.ylabel('fitness', fontsize=7)

    plt.show()

with plt.style.context(['science', 'ieee', 'grid']):
    plt.figure()
    plt.suptitle('Mejores fitness de Secuencias en cada Mutador', fontsize=7)
    plt.title(f'Secuencias Totales: {total_sequences}', fontsize=7)

    plt.plot(num_sequences_list, sm_best_fitness_list, linestyle='dashed')
    plt.plot(num_sequences_list, cm_best_fitness_list, linestyle='dashdot')
    plt.plot(num_sequences_list, cr_best_fitness_list, linestyle='dashed')
    plt.plot(num_sequences_list, sa_best_fitness_list, linestyle='dashdot')

    plt.xticks(num_sequences_list, fontsize=6)
    plt.yticks(y_ticks, fontsize=6)

    plt.xlabel('Número de Secuencias', fontsize=7)
    plt.ylabel('fitness', fontsize=7)

    plt.legend(['MSS', 'MSS + MCS', 'MSS + MRQ', 'MSS + RS'], fontsize=5)
    plt.show()

# TODO: graph the following combinations of mutators:
#   * crosser mutator + chemical reactions mutator
#   * simulated annealing + crosser mutator
#   * simulated annealing + chemical reactions mutator
#   * simulated annealing + crosser mutator + chemical reactions mutator
