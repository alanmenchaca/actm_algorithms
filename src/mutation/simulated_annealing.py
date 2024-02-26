from dataclasses import dataclass, field

import numpy as np

from utils.sequence import Sequence
from utils.fitness_calculator import FitnessCalculator
from mutation.simple_mutator import SimpleMutator


@dataclass
class SimulatedAnnealing:
    high_temperature: int = field(default=1000)
    final_temperature: float = field(default=0.01)
    cooling_rate: float = field(default=0.99)
    metropolis_criterion: float = field(default=100)

    # sequence with best genes alignment
    best_sequence: Sequence = field(default=None)

    _simple_mutator: SimpleMutator = field(default=None)
    _fitness_calculator: FitnessCalculator = field(default=None)

    def __post_init__(self):
        self._simple_mutator: SimpleMutator = SimpleMutator()
        self._fitness_calculator: FitnessCalculator = FitnessCalculator()

    def run_annealing(self, seq_to_compare: Sequence, sequence: Sequence) -> None:
        self.best_sequence = sequence
        self._fitness_calculator.set_seqA(seq_to_compare)
        self._mutate_and_compute_fitness(sequence)

        temperature: float = self.high_temperature
        while temperature > self.final_temperature:
            idx: int = 0
            while idx < self.metropolis_criterion:
                new_seq_mutated: Sequence = sequence.__copy__()
                self._mutate_and_compute_fitness(new_seq_mutated)
                self._simple_mutator.mutate_sequences_genes([new_seq_mutated])

                difference: float = new_seq_mutated.fitness - sequence.fitness

                if difference > 0:
                    # accept new solution
                    self.best_sequence = new_seq_mutated
                else:
                    probability: float = np.exp(-difference / temperature)
                    if probability > np.random.rand():
                        self.best_sequence = new_seq_mutated

                idx += 1

            temperature = self.cooling_rate * temperature
            # print(f'temperature: {round(temperature, 2)}')

    def get_best_sequence_found(self) -> Sequence:
        return self.best_sequence

    def _mutate_and_compute_fitness(self, sequence: Sequence) -> None:
        self._simple_mutator.mutate_sequences_genes([sequence])
        self._fitness_calculator.compute_fitness([sequence])
