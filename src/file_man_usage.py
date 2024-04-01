from mutation.simple_mutator import SimpleMutator
from utils.file_manager import SeqsSaver, SeqLoader
from utils.fitness_calculator import FitnessCalculator
from utils.sequence import Sequence

seq1: Sequence = SeqLoader.load("sequences/env_HIV1S.txt")
seq2: Sequence = SeqLoader.load("sequences/env_HIV1H.txt")
seq1.seq_id = "HIV1S"
seq2.seq_id = "HIV1H"

# seq1: Sequence = Sequence("A" * 60 + "C" * 10 + "-" * 10)
# seq2: Sequence = Sequence("B" * 60 + "C" * 10 + "-" * 10)

####################################################################

seq2_sm_population: list[Sequence] \
    = SimpleMutator.generate_mutated_seqs(seq2, 10)
FitnessCalculator.compute_seqs_fitness(seq1, seq2_sm_population)
best_seq: Sequence = seq2_sm_population[0]

SeqsSaver.save(seq1, best_seq)
