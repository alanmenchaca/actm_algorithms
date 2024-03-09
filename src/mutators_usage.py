from mutation.backpack_mutator import BackpackCrosserMutator
from mutation.chemical_reactions import ChemicalReactionsMutator
from mutation.simple_mutator import SimpleMutator
from mutation.simulated_annealing import SimulatedAnnealing
from utils.file_manager import SeqLoader
from utils.fitness_calculator import FitnessCalculator

from utils.sequence import Sequence

# seq1: Sequence = SeqLoader.load("sequences/env_HIV1S.txt")
# seq2: Sequence = SeqLoader.load("sequences/env_HIV1H.txt")

seq1: Sequence = Sequence("AAAA--DD")
seq2: Sequence = Sequence("BBBB--DD")

################################################################

sm: SimpleMutator = SimpleMutator()
seq1_sm_population: list[Sequence] = sm.generate_mutated_population(seq1, 3)
# seq2_sm_population: list[Sequence] = sm.generate_mutated_population(seq2, 10)

################################################################

# bcm: BackpackCrosserMutator = BackpackCrosserMutator()
# seq1_bcm_population: list[Sequence] = bcm.generate_mutated_population(seq1_sm_population)
# seq2_bcm_population: list[Sequence] = bcm.generate_mutated_population(seq2_sm_population)

################################################################

# crm: ChemicalReactionsMutator = ChemicalReactionsMutator()
# copy() is used to avoid changing the original population
# crm.collide_molecules(seq2, seq1_sm_population.copy(), 10)
# crm.collide_molecules(seq1, seq2_sm_population.copy(), 10)

################################################################

# sa: SimulatedAnnealing = SimulatedAnnealing()
# sa.run_annealing(seq2, seq1_sm_population[0])
# best_seq: Sequence = sa.get_best_sequence_found()

################################################################

# fc: FitnessCalculator = FitnessCalculator()
# fc.set_main_seq(seq2)
# fc.compute_fitness(seq1_sm_population)
# fc.compute_fitness(seq1_bcm_population)
# fc.compute_fitness([best_seq])

# best_seq = fc.get_best_seqs()["best_seq"]
# print(best_seq.fitness)

################################################################
