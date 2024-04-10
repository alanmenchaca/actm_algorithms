import time

import matplotlib.pyplot as plt

from mutation.chemical_reactions import ChemicalReactionsMutator as CRMutator
from mutation.simple_mutator import SimpleMutator
from utils.metrics import SeqsSimilarity
from utils.seq import Sequence
from utils.seqs_manager import SeqLoader

import scienceplots

# parameters:
#   * num_sequences_list
#   * num_collisions_list

seq1: Sequence = SeqLoader.load("../src/sequences/env_HIV1S.txt")
seq2: Sequence = SeqLoader.load("../src/sequences/env_HIV1H.txt")

num_seqs_list: list[int] = [50, 100, 150, 200]
num_collisions_list: list[int] = [5, 10, 15, 20, 30, 50]

# large collision dont improve the similarity
# num_collisions_list: list[int] = [100, 150, 200, 250, 300, 500]
# num_collisions_list: list[int] = [10, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500]

sm_best_similarities: list[float] = []
cr_best_similarities: list[float] = []

start_time = time.time()
# for idx, num_collisions in enumerate(num_collisions_list):
#     seq2_sm_seqs: list[Sequence] = SimpleMutator.generate_mutated_seqs(seq2, 100)
#     SeqsSimilarity.compute(seq1, seq2_sm_seqs)
#     sm_best_similarities.append(seq2_sm_seqs[0].similarity)
#
#     CRMutator.collide_molecules(seq1, seq2_sm_seqs, num_collisions)
#     SeqsSimilarity.compute(seq1, seq2_sm_seqs)
#     cr_best_similarities.append(seq2_sm_seqs[0].similarity)
#
#     print(f'{(idx + 1)}/{len(num_collisions_list)} collisions done!')

for idx, num_seqs in enumerate(num_seqs_list):
    seq2_sm_seqs: list[Sequence] = SimpleMutator.generate_mutated_seqs(seq2, num_seqs)
    SeqsSimilarity.compute(seq1, seq2_sm_seqs)
    sm_best_similarities.append(seq2_sm_seqs[0].similarity)

    CRMutator.collide_molecules(seq1, seq2_sm_seqs, 10)
    SeqsSimilarity.compute(seq1, seq2_sm_seqs)
    cr_best_similarities.append(seq2_sm_seqs[0].similarity)

    print(f'{(idx + 1)}/{len(num_seqs_list)} sequences done!')

end_time = time.time()
elapsed_time_minutes = (end_time - start_time) / 60
print(f'\nelapsed_time_minutes: {elapsed_time_minutes}')

print(f'sm_best_similarities: {sm_best_similarities}')
print(f'cr_best_similarities: {cr_best_similarities}')

# ################################################################################
# first plot to know what is the best_num_collisions for chemical reactions mutator
# plt.figure(figsize=(8, 6))
# with plt.style.context(['science', 'ieee', 'grid']):
#     plt.rcParams['font.size'] = 7
#
#     plt.suptitle('Rendimiento de Chemical Reactions Mutator', fontsize=7)
#     plt.title(f'Secuencias Totales: 100', fontsize=7)
#
#     plt.xlabel('Número de colisiones')
#     plt.ylabel('simmilaridad')
#
#     plt.plot(num_collisions_list, sm_best_similarities, label='Simple Mutator')
#     plt.plot(num_collisions_list, cr_best_similarities, label='CR Mutator')
#
#     plt.legend()
#     plt.show()

# ################################################################################

plt.plot(num_seqs_list, sm_best_similarities, marker='o', label='Simple Mutator')
plt.plot(num_seqs_list, cr_best_similarities, marker='o', label='Chemical Reactions Mutator')

plt.xlabel('Número de secuencias')
plt.ylabel('similaridad')

plt.suptitle('Rendimiento de los mutadores en función del número de secuencias')
plt.title('150 colisiones en el Mutador de Reacciones Químicas')

plt.legend()
plt.grid()
plt.show()
