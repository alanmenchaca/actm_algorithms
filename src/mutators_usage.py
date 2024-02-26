from mutation.simple_mutator import SimpleMutator
from utils.sequence import Sequence

# Download necessary
# Install necessary packages
# Import all packages
# Load seqA and seqB


# Simple Mutator implementation
# ------ simple mutator docu ------
# Simple Mutator
# < Simple mutator explanation >

# params:
#   - num_populations: number of populations to generate (default: 1000).
#   - num_sequences: number of sequences of each population (default: 100).
#   - rand_indexes_len: tuple with the range of the random indexes [min, max], default: [1, 6].
#   - gaps_lengths_arr: tuple with the range of the gaps lengths [min, max], default: [1, 3].

num_populations: int = 1000
num_sequences: int = 100
rand_indexes_len: tuple = (4, 8)
gaps_lengths_arr: tuple = (1, 2)

seqA: Sequence = Sequence("AABBCCDD")
seqB: Sequence = Sequence("AABBCCDD")

# sm.rand_indexes_len = (4, 8)
# sm.gaps_lengths_arr = (1, 2)
sm: SimpleMutator = SimpleMutator()
sm_mutated_seqs: list[Sequence] = sm.generate_mutated_population(seqA, 5)

print("\nSimpleMutator:")
for seq in sm_mutated_seqs:
    print(seq.genes)

# Backpack Mutator implementation
# ------ backpack mutator docu ------

# seqA: Sequence = Sequence("AABBCCDD")
# seqB: Sequence = Sequence("EEFFGGHH")
#
# bcm: BackpackCrosserMutator = BackpackCrosserMutator()
# bcm_mutated_seqs: list[Sequence] = bcm.generate_mutated_population([seqA])
#
# print("BackpackCrosserMutator:")
# for seq in bcm_mutated_seqs:
#     print(seq)

# print("--------------------------------------------------------")

# seqA: Sequence = Sequence("AABBCCDD")
# seqB: Sequence = Sequence("EEFFGGHH")
#
# crm: ChemicalReactionsMutator = ChemicalReactionsMutator()
# crm_mutated_seqs: list[Sequence] = crm.collide_molecules([seqA])

# print("--------------------------------------------------------")
