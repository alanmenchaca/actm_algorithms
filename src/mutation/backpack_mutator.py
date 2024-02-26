from dataclasses import dataclass, field
from typing import Iterator

import numpy as np
from exceptions.sequence import SequenceError
from numpy import ndarray
from utils.sequence import Sequence


@dataclass(init=False, repr=False)
class BackpackCrosserMutator:
    _population: list[Sequence] = field(default_factory=list)
    _seqA: Sequence = field(default_factory=Sequence)
    _seqB: Sequence = field(default_factory=Sequence)

    @SequenceError.validate_sequences
    def generate_mutated_population(self, sequences: list[Sequence]) -> list[Sequence]:
        self._append_seqA_to_sequences_if_sequences_len_is_odd(sequences)
        sequences_iter: Iterator[Sequence] = iter(sequences)
        self._population: list[Sequence] = []

        for sequence in sequences_iter:
            self._seqA: Sequence = sequence.__copy__()
            self._seqB: Sequence = next(sequences_iter).__copy__()

            self._swap_genes_between_seqA_and_seqB()
            self._append_new_sequences_to_population()

        return self._population

    @staticmethod
    def _append_seqA_to_sequences_if_sequences_len_is_odd(sequences: list[Sequence]) -> None:
        if len(sequences) % 2 != 0:
            sequences.append(sequences[0])

    def _swap_genes_between_seqA_and_seqB(self) -> None:
        new_seqA_arr, new_seqB_arr = self._generate_new_seqA_and_seqB_arr_with_swapped_genes()
        self._assign_new_genes_arr_to_seqA_and_seqB(new_seqA_arr, new_seqB_arr)

    def _generate_new_seqA_and_seqB_arr_with_swapped_genes(self) -> tuple[ndarray, ndarray]:
        cross_point: int = np.random.randint(1, len(self._seqA.get_indexes_of_genes()) - 1)
        first_arr_seq_a_cp, second_arr_seq_a_cp = self._get_seq_cross_points(self._seqA, cross_point)
        first_arr_seq_b_cp, second_arr_seq_b_cp = self._get_seq_cross_points(self._seqB, cross_point)

        new_first_arr: ndarray = np.concatenate(
            (self._seqA.genes_as_arr[:first_arr_seq_a_cp],
             self._seqB.genes_as_arr[first_arr_seq_b_cp:])
        )
        new_second_arr: ndarray = np.concatenate(
            (self._seqB.genes_as_arr[:second_arr_seq_b_cp],
             self._seqA.genes_as_arr[second_arr_seq_a_cp:])
        )

        return new_first_arr, new_second_arr

    @staticmethod
    def _get_seq_cross_points(seq: Sequence, cross_point: int) -> tuple[int, int]:
        arr_first_cp: int = seq.get_indexes_of_genes()[cross_point]
        arr_second_cp: int = seq.get_indexes_of_genes()[-cross_point]
        return arr_first_cp, arr_second_cp

    def _assign_new_genes_arr_to_seqA_and_seqB(self, new_seqA_arr: ndarray,
                                               new_seqB_arr: ndarray) -> None:
        self._seqA.genes_as_arr = new_seqA_arr
        self._seqB.genes_as_arr = new_seqB_arr

    def _append_new_sequences_to_population(self) -> None:
        self._population.append(self._seqA)
        self._population.append(self._seqB)
