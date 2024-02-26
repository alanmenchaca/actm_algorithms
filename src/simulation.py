from dataclasses import dataclass, field
from typing import Callable

from tqdm import tqdm

from utils.sequence import Sequence
from utils.fitness_calculator import FitnessCalculator
from mutation.backpack_mutator import BackpackCrosserMutator
from mutation.chemical_reactions import ChemicalReactionsMutator
from mutation.simple_mutator import SimpleMutator
from mutation.simulated_annealing import SimulatedAnnealing
from utils.file_manager import TxtFileReader


@dataclass(repr=False)
class Simulation:
    seqA: Sequence = field(init=False, default=None)
    seqB: Sequence = field(init=False, default=None)

    _seqA_list: list[Sequence] = field(init=False, default=None)
    _seqB_list: list[Sequence] = field(init=False, default=None)

    _num_populations: int = field(init=False, default=100)
    _num_sequences_per_population: int = field(init=False, default=10)

    _simple_mutator: SimpleMutator = field(init=False, default_factory=SimpleMutator)
    _crosser_mutator: BackpackCrosserMutator = field(init=False, default_factory=BackpackCrosserMutator)
    _cr_mutator: ChemicalReactionsMutator = field(init=False, default_factory=ChemicalReactionsMutator)
    _annealing_mutator: SimulatedAnnealing = field(init=False, default_factory=SimulatedAnnealing)

    _fitness: FitnessCalculator = field(init=False, default_factory=FitnessCalculator)

    def __post_init__(self):
        self._num_populations: int = int(input("Ingresa el número de poblaciones: "))
        self._num_sequences_per_population: int = int(input("Ingresa el número de secuencias por población: "))

        self._initialize_main_and_secondary_sequences()
        self._fitness.set_seqA(self.seqA)

    def _initialize_main_and_secondary_sequences(self) -> None:
        seqA: str = TxtFileReader.read('./genes/env_HIV1H.txt')
        seqB: str = TxtFileReader.read('./genes/env_HIV1S.txt')

        self.seqA: Sequence = Sequence(genes=seqA)
        self.seqB: Sequence = Sequence(genes=seqB)

    def run_simulation(self):
        generate_population_func_list: list[Callable] = [
            self._generate_population_from_seqA_and_seqB_using_simple_mutator,
            self._generate_population_from_seqA_and_seqB_using_backpack_crosser_mutator
        ]

        tdqm_config = {'desc': 'Running simulation', 'unit': 'population', 'ncols': 100}
        for _ in tqdm(range(self._num_populations), **tdqm_config):
            for generate_population_func in generate_population_func_list:
                generate_population_func()

                for sequence in self._seqA_list:
                    self._fitness.set_seqA(sequence)
                    self._fitness.compute_fitness(self._seqB_list)

        self.run_chemical_reactions_mutator()
        # for sequence in self._seqA_list:
        #     self._fitness.set_seqA(sequence)
        #     self._fitness.compute_fitness(self._seqB_list)

        print(f"\nPrimera secuencia: {self.seqA}")
        print(f"Segunda secuencia: {self.seqB}")

        best_sequences: dict[str, Sequence] = self._fitness \
            .get_best_sequences_of_each_populations()
        print(f"\nMejor secuencia (env_HIV1H): {best_sequences['seqA']}")
        print(f"Mejor secuencia (env_HIV1S): {best_sequences['seqB']}")

        # seqA, seqB = best_sequences
        # headers: str = "Secuencia A\tSecuencia B\tFitness"
        # TxtFileSaver.format_and_save_sequences(seqA, seqB, "formatted_sequences.txt")

    def _generate_population_from_seqA_and_seqB_using_simple_mutator(self) -> None:
        self._seqA_list = self._simple_mutator.generate_mutated_population(
            sequence=self.seqA, num_sequences=self._num_sequences_per_population)
        self._seqB_list = self._simple_mutator.generate_mutated_population(
            sequence=self.seqB, num_sequences=self._num_sequences_per_population)

    def _generate_population_from_seqA_and_seqB_using_backpack_crosser_mutator(self) -> None:
        self._seqA_list = self._crosser_mutator. \
            generate_mutated_population(sequences=self._seqA_list)
        self._seqB_list = self._crosser_mutator \
            .generate_mutated_population(sequences=self._seqB_list)

    def run_chemical_reactions_mutator(self):
        pass


if __name__ == '__main__':
    Simulation().run_simulation()
