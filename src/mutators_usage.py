from mutation.backpack_mutator import BackpackCrosserMutator
from mutation.chemical_reactions import ChemicalReactionsMutator
from mutation.simple_mutator import SimpleMutator
from mutation.simulated_annealing import SimulatedAnnealing
from utils.file_manager import TxtFileReader
from utils.fitness_calculator import FitnessCalculator
from utils.sequence import Sequence

seqA: Sequence = Sequence("AABBCCDD")
seqB: Sequence = Sequence("EEFFGGHH")

sm: SimpleMutator = SimpleMutator()
bcm: BackpackCrosserMutator = BackpackCrosserMutator()
crm: ChemicalReactionsMutator = ChemicalReactionsMutator()
sa: SimulatedAnnealing = SimulatedAnnealing()

fc: FitnessCalculator = FitnessCalculator()
file_reader: TxtFileReader = TxtFileReader()

sm.rand_indexes_len = (4, 8)
sm.gaps_lengths_arr = (1, 2)
sm_mutated_seqs: list[Sequence] = sm.generate_mutated_population(seqA, 100)
bcm_mutated_seqs: list[Sequence] = bcm.generate_mutated_population(sm_mutated_seqs)

# crm_mutated_seqs: list[Sequence] = crm.collide_molecules(bcm_mutated_seqs)
