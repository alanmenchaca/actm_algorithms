from mutation.crosser_mutator import CrosserMutator
from utils.seq import Sequence

seq1: Sequence = Sequence("ATTGA---GC-CTG--TATCAGC--CC")
seq2: Sequence = Sequence("ATT--GAGC--CTG-TATC--AGCC--C")

sm_seqs: list[Sequence] = [seq1, seq2]
print("\nsm_seqs")
for seq in sm_seqs:
    print(seq.genes)
print()

cm_seqs: list[Sequence] = (CrosserMutator
                           .generate_mutated_seqs(sm_seqs))

# sm_seqs
# ATTGA---GC-CTG--TATCAGC--CC
# ATT--GAGC--CTG-TATC--AGCC--C

# cp: 11
# ATTGA---GC-CTG--T  ATC--AGCC--C
# ATT--GAGC--C  TG--TATCAGC--CC
