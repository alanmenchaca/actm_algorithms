from mutation.chemical_reactions import ChemicalReactionsMutator as CRMutator
from mutation.crosser_mutator import CrosserMutator
from mutation.simple_mutator import SimpleMutator
from mutation.simulated_annealing import SimulatedAnnealing
from utils.metrics import SeqsSimilarity
from utils.seq import Sequence
from utils.seqs_manager import SeqLoader, SeqsSaver

# seq1: Sequence = SeqLoader.load("./sequences/env_HIV1H.txt")
# seq2: Sequence = SeqLoader.load("./sequences/env_HIV1S.txt")
# seq1.seq_id, seq2.seq_id = "HIV1S ", "HIV1H "

seq1: Sequence = Sequence("AAADDDCCC")
seq2: Sequence = Sequence("AAADDDCCC")

num_seqs: int = 10
populations: int = 1

################################################################

for i in range(populations):
    # print(f"\nPopulation: {(i + 1)} - seq1[{seq1.genes}]")
    SimpleMutator.set_params((1, 6), (1, 3))
    seq2_sm_seqs: list[Sequence] = SimpleMutator.generate_mutated_seqs(seq2, num_seqs)
    SeqsSimilarity.compute(seq1, seq2_sm_seqs)

    # for sm_seq in seq2_sm_seqs:
    #     print(sm_seq.seq_id, sm_seq.get_genes_without_mutations(),
    #           sm_seq.similarity, sm_seq.genes)

################################################################

for i in range(populations):
    print(f"\nPopulation: {(i + 1)} - seq1[{seq1.genes}]")
    seq2_sm_seqs: list[Sequence] = SimpleMutator.generate_mutated_seqs(seq2, num_seqs)
    seq2_cm_seqs: list[Sequence] = CrosserMutator.generate_mutated_seqs(seq2_sm_seqs)
    SeqsSimilarity.compute(seq1, seq2_cm_seqs)

    # for sm_seq, cm_seq in zip(seq2_sm_seqs, seq2_cm_seqs):
    #    print(cm_seq.seq_id, cm_seq.get_genes_without_mutations(),
    #          cm_seq.similarity, sm_seq.genes, cm_seq.genes)

################################################################

for i in range(populations):
    # print(f"\nPopulation: {(i + 1)} - seq1[{seq1.genes}]")
    seq2_sm_seqs: list[Sequence] = SimpleMutator.generate_mutated_seqs(seq2, num_seqs)
    SeqsSimilarity.compute(seq1, seq2_sm_seqs)

    # for sm_seq in seq2_sm_seqs:
    #     print(sm_seq.seq_id, sm_seq.get_genes_without_mutations(),
    #           sm_seq.similarity, sm_seq.genes)

    CRMutator.collide_molecules(seq1, seq2_sm_seqs, 3)
    SeqsSimilarity.compute(seq1, seq2_sm_seqs)

    # print()
    # for sm_seq in seq2_sm_seqs:
    #     print(sm_seq.seq_id, sm_seq.get_genes_without_mutations(),
    #           sm_seq.similarity, sm_seq.genes)

################################################################

# best_seq: Sequence = SimulatedAnnealing.run(seq1, seq2)
# SeqsSaver.save(seq1, seq2)
# print(best_seq)
