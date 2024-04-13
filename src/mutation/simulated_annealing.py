from dataclasses import dataclass
from typing import ClassVar

import numpy as np

from mutation.simple_mutator import SimpleMutator
from utils.metrics import SeqsSimilarity
from utils.seq import Sequence


@dataclass
class SimulatedAnnealing:
    _seq_ref: ClassVar[Sequence] = None
    _seq: ClassVar[Sequence] = None
    _best_seq: ClassVar[Sequence] = None

    _high_temperature: ClassVar[int] = 1000
    _final_temperature: ClassVar[float] = 0.01
    _cooling_rate: ClassVar[float] = 0.99
    _metropolis_criterion: ClassVar[int] = 100

    _temperature: ClassVar[float] = 0.0

    @classmethod
    def set_params(cls, high_temperature: int, final_temperature: float,
                   cooling_rate: float, metropolis_criterion: int, ) -> None:
        cls._high_temperature = high_temperature
        cls._final_temperature = final_temperature
        cls._cooling_rate = cooling_rate
        cls._metropolis_criterion = metropolis_criterion

    @classmethod
    def run(cls, main_seq: Sequence, seq: Sequence) -> Sequence:
        cls._init_seqs(main_seq, seq)
        cls._execute_algorithm()
        return cls._best_seq

    @classmethod
    def _init_seqs(cls, seq_ref: Sequence, seq_to_mutate: Sequence) -> None:
        cls._seq_ref = seq_ref
        cls._best_seq = seq_to_mutate
        cls._seq = seq_to_mutate

    @classmethod
    def _execute_algorithm(cls) -> None:
        cls._temperature = cls._high_temperature

        while cls._temperature > cls._final_temperature:
            cls._run_sa_until_metropolis_criterion()
            print(f'temperature: {round(cls._temperature, 2)}')

    @classmethod
    def _run_sa_until_metropolis_criterion(cls) -> None:
        for _ in range(cls._metropolis_criterion):
            seq_mutated: Sequence = cls._generate_seq_mutated()
            difference: float = seq_mutated.similarity - cls._seq.similarity

            if difference > 0:
                cls._best_seq = seq_mutated
            else:
                probability: float = np.exp(-difference / cls._temperature)
                if probability > np.random.rand():
                    cls._best_seq = seq_mutated

        cls._decrease_temperature()

    @classmethod
    def _generate_seq_mutated(cls) -> Sequence:
        seq_mutated: Sequence = cls._seq.__copy__()
        SimpleMutator.mutate_seqs_genes([seq_mutated])
        SeqsSimilarity.compute(cls._seq_ref, [seq_mutated])
        return seq_mutated

    @classmethod
    def _decrease_temperature(cls) -> None:
        cls._temperature = cls._temperature * cls._cooling_rate
