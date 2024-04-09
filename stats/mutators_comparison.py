import time

import matplotlib.pyplot as plt
import numpy as np

from mutation.chemical_reactions import ChemicalReactionsMutator as CRMutator
from mutation.crosser_mutator import CrosserMutator
from mutation.simple_mutator import SimpleMutator
from mutation.simulated_annealing import SimulatedAnnealing
from utils.file_manager import SeqLoader
from utils.fitness_calculator import FitnessCalculator
from utils.sequence import Sequence

import scienceplots

seq1: Sequence = SeqLoader.load("../src/sequences/env_HIV1S.txt")
seq2: Sequence = SeqLoader.load("../src/sequences/env_HIV1H.txt")

# num_sequences_list: list[int] = [10, 20, 30, 40, 50]
num_sequences_list: list[int] = [200, 300, 400, 500, 600, 700, 800]
num_collisions: int = 100
total_sequences: int = 0

sm_best_fitness_list: list[float] = []  # simple mutator
cm_best_fitness_list: list[float] = []  # crosser mutator
cr_best_fitness_list: list[float] = []  # chemical reactions mutator
sa_best_fitness_list: list[float] = []  # simulated annealing

start_time = time.time()
for idx, num_seqs in enumerate(num_sequences_list):
    seq2_sm_seqs: list[Sequence] = SimpleMutator.generate_mutated_seqs(seq2, num_seqs)
    FitnessCalculator.compute_seqs_fitness(seq1, seq2_sm_seqs)
    sm_best_fitness_list.append(seq2_sm_seqs[0].fitness)

    seq2_cm_seqs: list[Sequence] = CrosserMutator.generate_mutated_seqs(seq2_sm_seqs)
    FitnessCalculator.compute_seqs_fitness(seq1, seq2_cm_seqs)
    cm_best_fitness_list.append(seq2_cm_seqs[0].fitness)

    # a copy of seq2_sm_seqs is passed to avoid modifying the original list
    seq2_sm_seqs_copy: list[Sequence] = seq2_sm_seqs.copy()
    CRMutator.collide_molecules(seq1, seq2_sm_seqs_copy, 10)
    FitnessCalculator.compute_seqs_fitness(seq1, seq2_sm_seqs_copy)
    cr_best_fitness_list.append(seq2_sm_seqs_copy[0].fitness)

    best_seq: Sequence = SimulatedAnnealing.run_annealing(seq1.__copy__(), seq2.__copy__())
    sa_best_fitness_list.append(best_seq.fitness)
    # sa_best_fitness_list.append(np.random.randint(0, 1000))

    total_sequences += len(seq2_sm_seqs) + len(seq2_cm_seqs) + len(seq2_sm_seqs_copy)
    print(f'total_sequences: {total_sequences}')
    print(f'num. sequences completed: {(idx + 1)}/{len(num_sequences_list)}\n')

end_time = time.time()
elapsed_time_minutes = (end_time - start_time) / 60
print(f'\nelapsed_time_minutes: {elapsed_time_minutes}')

print(f'sm_best_fitness_list: {sm_best_fitness_list}')
print(f'cm_best_fitness_list: {cm_best_fitness_list}')
print(f'cr_best_fitness_list: {cr_best_fitness_list}')
print(f'sa_best_fitness_list: {sa_best_fitness_list}')


def round_to_nearest_multiple(value: int, base: int = 100) -> int:
    remainder: int = value % base
    if remainder <= (base / 2):
        return value - remainder
    else:
        return value + (base - remainder)


all_best_fitness = np.concatenate((sm_best_fitness_list, cm_best_fitness_list,
                                   cr_best_fitness_list, sa_best_fitness_list))
min_nearest_multiple = round_to_nearest_multiple(min(all_best_fitness))
max_nearest_multiple = round_to_nearest_multiple(max(all_best_fitness))
y_ticks = np.arange(min_nearest_multiple, (max_nearest_multiple + 100), 100)

# ################################################################################

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

# ################################################################################

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

    plt.legend(['MSS', 'MSS + MCS', 'MSS + MRQ', 'RS'], fontsize=5)
    plt.show()

# TODO: graph the following combinations of mutators:
#   * crosser mutator + chemical reactions mutator
#   * simulated annealing + crosser mutator
#   * simulated annealing + chemical reactions mutator
#   * simulated annealing + crosser mutator + chemical reactions mutator
