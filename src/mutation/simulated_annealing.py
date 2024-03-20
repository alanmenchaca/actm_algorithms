from dataclasses import dataclass
from typing import ClassVar

import numpy as np

from mutation.simple_mutator import SimpleMutator
from utils.fitness_calculator import FitnessCalculator
from utils.sequence import Sequence


# TODO: REFACTOR THIS CLASS
@dataclass
class SimulatedAnnealing:
    high_temperature: ClassVar[int] = 1000
    final_temperature: ClassVar[float] = 0.01
    cooling_rate: ClassVar[float] = 0.99
    metropolis_criterion: ClassVar[int] = 100

    _seq_mutated: ClassVar[Sequence] = None
    _best_seq: ClassVar[Sequence] = None

    @classmethod
    def run_annealing(cls, main_seq: Sequence, seq: Sequence) -> Sequence:
        cls._best_seq = seq
        cls._mutate_seq_and_compute_fitness(main_seq, seq)
        temperature: float = cls.high_temperature

        while temperature > cls.final_temperature:
            for _ in range(cls.metropolis_criterion):
                cls._seq_mutated: Sequence = seq.__copy__()
                cls._mutate_seq_and_compute_fitness(main_seq, cls._seq_mutated)
                difference: float = cls._seq_mutated.fitness - seq.fitness

                if difference > 0:
                    cls._best_seq = cls._seq_mutated
                else:
                    probability: float = np.exp(-difference / temperature)
                    if probability > np.random.rand():
                        cls._best_seq = cls._seq_mutated

            temperature = cls.cooling_rate * temperature
            print(f'temperature: {round(temperature, 2)}')

        return cls._best_seq

    @classmethod
    def _mutate_seq_and_compute_fitness(cls, seq_to_compare: Sequence, seq: Sequence) -> None:
        SimpleMutator.mutate_seqs_genes([seq])
        FitnessCalculator.compute_seqs_fitness(seq_to_compare, [seq])
