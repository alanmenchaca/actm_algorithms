from mutation.simple_mutator import SimpleMutator
from utils.file_manager import SeqsSaver, SeqLoader
from utils.fitness_calculator import FitnessCalculator
from utils.sequence import Sequence

seq1: Sequence = SeqLoader.load("sequences/env_HIV1S.txt")
seq2: Sequence = SeqLoader.load("sequences/env_HIV1H.txt")

# seq1: Sequence = Sequence("A" * 60 + "C" * 10 + "-" * 10)
# seq2: Sequence = Sequence("B" * 60 + "C" * 10 + "-" * 10)

seq1.seq_id = "HIV1S"
seq2.seq_id = "HIV1H"

####################################################################

sm: SimpleMutator = SimpleMutator()
seq1_sm_population: list[Sequence] = sm.generate_mutated_population(seq1, 10)

####################################################################

fc: FitnessCalculator = FitnessCalculator()
fc.set_main_seq(seq2)
fc.compute_fitness(seq1_sm_population)
best_seq = fc.get_best_seqs()["best_seq"]

SeqsSaver.save(seq2, best_seq)
print(seq2)
print(best_seq)
