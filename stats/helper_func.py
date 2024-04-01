import numpy as np

from utils.sequence import Sequence
from utils.fitness_calculator import FitnessCalculator
from mutation.backpack_mutator import BackpackCrosserMutator
from mutation.chemical_reactions import ChemicalReactionsMutator
from mutation.simple_mutator import SimpleMutator
from mutation.simulated_annealing import SimulatedAnnealing
from utils.file_manager import TxtFileReader

simple_mutator: SimpleMutator = SimpleMutator()
crosser_mutator: BackpackCrosserMutator = BackpackCrosserMutator()
cr_mutator: ChemicalReactionsMutator = ChemicalReactionsMutator()
simulated_annealing: SimulatedAnnealing = SimulatedAnnealing()

fitness_calculator: FitnessCalculator = FitnessCalculator()


def load_seqA_and_seqB() -> tuple[Sequence, Sequence]:
    seqA: Sequence = Sequence(TxtFileReader.read('../src/sequences/env_HIV1H.txt'))
    seqB: Sequence = Sequence(TxtFileReader.read('../src/sequences/env_HIV1S.txt'))
    return seqA, seqB


def generate_populations_of_cr_mutator(seqA_population: list[Sequence],
                                       seqB_population: list[Sequence],
                                       num_collisions: int) -> tuple[list[Sequence], list[Sequence]]:
    seqA_cr_population = seqA_population.copy()
    seqB_cr_population = seqB_population.copy()

    # remove the rand_seq_to_compare
    rand_seq_to_compare = seqA_population[np.random.randint(0, len(seqA_cr_population))]
    seqA_cr_population.remove(rand_seq_to_compare)

    cr_mutator.collide_molecules(rand_seq_to_compare, seqA_cr_population, num_collisions)
    cr_mutator.collide_molecules(rand_seq_to_compare, seqB_cr_population, num_collisions)
    return seqA_cr_population, seqB_cr_population


def generate_populations_of_crosser_mutator(seqA_population: list[Sequence],
                                            seqB_population: list[Sequence]) \
        -> tuple[list[Sequence], list[Sequence]]:
    seqA_population: list[Sequence] = crosser_mutator.generate_mutated_seqs(seqA_population)
    seqB_population: list[Sequence] = crosser_mutator.generate_mutated_seqs(seqB_population)
    return seqA_population, seqB_population


def generate_populations_of_simple_mutator(seqA: Sequence, seqB: Sequence,
                                           num_sequences: int, params: dict[str, tuple]) \
        -> tuple[list[Sequence], list[Sequence]]:
    if params:
        simple_mutator._rand_indexes_len = params['rand_indexes_len']
        simple_mutator.set_gaps_lengths_arr = params['gaps_lengths_arr']

    seqA_population: list[Sequence] = simple_mutator.generate_mutated_seqs(seqA, num_sequences // 2)
    seqB_population: list[Sequence] = simple_mutator.generate_mutated_seqs(seqB, num_sequences // 2)
    return seqA_population, seqB_population


def run_simulated_annealing(population: list[Sequence]) -> Sequence:
    rand_seqA = population[np.random.randint(0, len(population))]
    rand_seqB = population[np.random.randint(0, len(population))]

    simulated_annealing.run_annealing(rand_seqA, rand_seqB)
    return simulated_annealing.get_best_sequence_found()


def append_best_fitness_from_populations_to_list(seqA_population: list[Sequence],
                                                 seqB_population: list[Sequence],
                                                 best_fitness_list: list[float]) -> None:
    best_seqA, best_seqB = get_best_sequences_from_populations(seqA_population, seqB_population)
    best_fitness_list.append(best_seqA.fitness)
    reset_fitness_calculator_params()


def get_best_sequences_from_populations(seqA_population: list[Sequence],
                                        seqB_population: list[Sequence]) -> tuple[Sequence, Sequence]:
    for seqA_to_compare in seqA_population:
        fitness_calculator.set_main_seq(seqA_to_compare)
        fitness_calculator.compute_seqs_fitness(seqB_population)

    best_sequences = fitness_calculator.get_best_seq_of_each_population()
    return best_sequences['seqA'], best_sequences['seqB']


def reset_fitness_calculator_params() -> None:
    fitness_calculator._best_seq = None
    fitness_calculator._best_seqs = None


def round_to_nearest_multiple(value: int, base: int = 100) -> int:
    remainder: int = value % base
    if remainder <= base / 2:
        return value - remainder
    else:
        return value + (base - remainder)
