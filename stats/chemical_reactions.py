import time

import matplotlib.pyplot as plt
import scienceplots

from actm_algorithms.stats.helper_func import load_seqA_and_seqB, \
    generate_populations_of_simple_mutator, append_best_fitness_from_populations_to_list, \
    generate_populations_of_cr_mutator
from mutation.chemical_reactions import ChemicalReactionsMutator
from mutation.simple_mutator import SimpleMutator

# parameters:
#   * num_sequences_list
#   * num_collisions_list

simple_mutator: SimpleMutator = SimpleMutator()
cr_mutator: ChemicalReactionsMutator = ChemicalReactionsMutator()

seqA, seqB = load_seqA_and_seqB()
num_sequences_list: list[int] = [50, 100, 150, 200]
# num_collisions_list: list[int] = [10, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500]
num_collisions_list: list[int] = [5, 10, 15, 20, 30, 50]

# large collision dont improve the fitness
# num_collisions_list: list[int] = [100, 150, 200, 250, 300, 500]

sm_best_fitness_list: list[float] = []
cr_best_fitness_list: list[float] = []

start_time = time.time()
# for idx, num_collisions in enumerate(num_collisions_list):
#     seqA_sm_population, seqB_sm_population = \
#         generate_populations_of_simple_mutator(seqA, seqB, 100, {})
#     append_best_fitness_from_populations_to_list(seqA_sm_population, seqB_sm_population,
#                                                  sm_best_fitness_list)
#
#     seqA_cr_population, seqB_cr_population = \
#         generate_populations_of_cr_mutator(seqA_sm_population, seqB_sm_population, num_collisions)
#     append_best_fitness_from_populations_to_list(seqA_cr_population, seqB_cr_population,
#                                                  cr_best_fitness_list)
#     print(f'{(idx + 1)}/{len(num_collisions_list)} collisions done!')

for idx, num_sequences in enumerate(num_sequences_list):
    seqA_sm_population, seqB_sm_population = \
        generate_populations_of_simple_mutator(seqA, seqB, num_sequences, {})
    append_best_fitness_from_populations_to_list(seqA_sm_population, seqB_sm_population,
                                                 sm_best_fitness_list)

    seqA_cr_population, seqB_cr_population = \
        generate_populations_of_cr_mutator(seqA_sm_population, seqB_sm_population, 10)
    append_best_fitness_from_populations_to_list(seqA_cr_population, seqB_cr_population,
                                                 cr_best_fitness_list)
    print(f'{(idx + 1)}/{len(num_sequences_list)} sequences done!')

end_time = time.time()
elapsed_time_minutes = (end_time - start_time) / 60
print(f'\nelapsed_time_minutes: {elapsed_time_minutes}')

print(f'sm_best_fitness_list: {sm_best_fitness_list}')
print(f'cr_best_fitness_list: {cr_best_fitness_list}')

# ################################################################################
# first plot to know what is the best_num_collisions for chemical reactions mutator

# with plt.style.context(['science', 'ieee', 'grid']):
#     plt.figure()
#     plt.rcParams['font.size'] = 7
#
#     plt.suptitle('Rendimiento de Chemical Reactions Mutator', fontsize=7)
#     plt.title(f'Secuencias Totales: 100', fontsize=7)
#
#     plt.xlabel('Número de colisiones')
#     plt.ylabel('fitness')
#
#     plt.plot(num_collisions_list, sm_best_fitness_list, label='Simple Mutator')
#     plt.plot(num_collisions_list, cr_best_fitness_list, label='CR Mutator')
#
#     plt.legend()
#     plt.show()

# ################################################################################

plt.plot(num_sequences_list, sm_best_fitness_list, marker='o', label='Simple Mutator')
plt.plot(num_sequences_list, cr_best_fitness_list, marker='o', label='Chemical Reactions Mutator')

plt.xlabel('Número de secuencias')
plt.ylabel('fitness')

plt.suptitle('Rendimiento de los mutadores en función del número de secuencias')
plt.title('150 colisiones en el Mutador de Reacciones Químicas')

plt.legend()
plt.grid()
plt.show()
