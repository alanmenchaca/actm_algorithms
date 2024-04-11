from mutation.simple_mutator import SimpleMutator
from utils.metrics import SeqsSimilarity
from utils.seq import Sequence
from utils.seqs_manager import SeqLoader

seq1: Sequence = SeqLoader.load("./src/sequences/env_HIV1H.txt")
seq2: Sequence = SeqLoader.load("./src/sequences/env_HIV1S.txt")

num_seqs: int = 1000
populations: int = 10
best_seqs = []

for population in range(populations):
    seq2_sm_seqs: list[Sequence] = SimpleMutator.generate_mutated_seqs(seq2, num_seqs)
    SeqsSimilarity.compute(seq1, seq2_sm_seqs)

    print(seq2_sm_seqs[0].similarity)
    best_seqs.append(seq2_sm_seqs[0])
    print('-' * 54)

best_seqs.sort(key=lambda seq: seq.similarity, reverse=True)
for idx, seq in enumerate(best_seqs[:10]):
    print(f"   {(idx + 1):02d}. * seq_id: {seq.seq_id} * similarity: {seq.similarity}")
