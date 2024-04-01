from mutation.backpack_mutator import BackpackCrosserMutator as BCMutator
from mutation.chemical_reactions import ChemicalReactionsMutator as CRMutator
from mutation.simple_mutator import SimpleMutator
# from mutation.simulated_annealing import SimulatedAnnealing
from utils.file_manager import SeqLoader, SeqsSaver
from utils.fitness_calculator import FitnessCalculator
from utils.sequence import Sequence

seq1: Sequence = SeqLoader.load("sequences/env_HIV1S.txt")
seq2: Sequence = SeqLoader.load("sequences/env_HIV1H.txt")
seq1.seq_id, seq2.seq_id = "HIV1S ", "HIV1H "

# seq1: Sequence = Sequence("AAADDDCCC")
# seq2: Sequence = Sequence("AAADDDCCC")

num_seqs: int = 10
populations: int = 10

################################################################

for i in range(populations):
    # print(f"\nPopulation: {(i + 1)} - seq1[{seq1.genes}]")
    SimpleMutator.set_params((1, 6), (1, 3))
    seq2_sm_seqs: list[Sequence] = SimpleMutator.generate_mutated_seqs(seq2, num_seqs)
    FitnessCalculator.compute_seqs_fitness(seq1, seq2_sm_seqs)

    # for sm_seq in seq2_sm_seqs:
    #     print(sm_seq.seq_id, sm_seq.get_genes_without_mutations(),
    #           sm_seq.fitness, sm_seq.genes)

################################################################

for i in range(populations):
    # print(f"\nPopulation: {(i + 1)} - seq1[{seq1.genes}]")
    seq2_sm_seqs: list[Sequence] = SimpleMutator.generate_mutated_seqs(seq2, num_seqs)
    seq2_bcm_seqs: list[Sequence] = BCMutator.generate_mutated_seqs(seq2_sm_seqs)
    FitnessCalculator.compute_seqs_fitness(seq1, seq2_bcm_seqs)

    # for sm_seq, bcm_seq in zip(seq2_sm_seqs, seq2_bcm_seqs):
    #     print(bcm_seq.seq_id, bcm_seq.get_genes_without_mutations(),
    #           bcm_seq.fitness, sm_seq.genes, bcm_seq.genes)

################################################################

for i in range(populations):
    # print(f"\nPopulation: {(i + 1)} - seq1[{seq1.genes}]")
    seq2_sm_seqs: list[Sequence] = SimpleMutator.generate_mutated_seqs(seq2, num_seqs)
    FitnessCalculator.compute_seqs_fitness(seq1, seq2_sm_seqs)

    # for sm_seq in seq2_sm_seqs:
    #     print(sm_seq.seq_id, sm_seq.get_genes_without_mutations(),
    #           sm_seq.fitness, sm_seq.genes)

    CRMutator.collide_molecules(seq1, seq2_sm_seqs, 5)
    FitnessCalculator.compute_seqs_fitness(seq1, seq2_sm_seqs)

    # print()
    # for sm_seq in seq2_sm_seqs:
    #     print(sm_seq.seq_id, sm_seq.get_genes_without_mutations(),
    #           sm_seq.fitness, sm_seq.genes)

################################################################

# best_seq: Sequence = SimulatedAnnealing.run_annealing(seq1, seq2)
# print(best_seq)

SeqsSaver.save(seq1, seq2)
