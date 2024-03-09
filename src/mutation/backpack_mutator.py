from dataclasses import dataclass, field
from typing import Iterator

import numpy as np
from numpy import ndarray

from exceptions.sequence import SeqsValidator as sv
from utils.sequence import Sequence


@dataclass(init=False, repr=False)
class BackpackCrosserMutator:
    _population: list[Sequence] = field(default=list)
    _seq1: Sequence = field(default=Sequence)
    _seq2: Sequence = field(default=Sequence)

    def generate_mutated_population(self, sequences: list[Sequence]) -> list[Sequence]:
        sv.validate_seqs(sequences)
        self._append_seq1_to_seqs_if_seqs_len_is_odd(sequences)
        seqs_iter: Iterator[Sequence] = iter(sequences)
        self._population: list[Sequence] = []
        idx: range = range(1, len(sequences), 2)

        for i, seq in zip(idx, seqs_iter):
            self._seq1: Sequence = seq.__copy__()
            self._seq2: Sequence = next(seqs_iter).__copy__()

            self._append_backpack_mutator_id_to_seqs(i)
            self._swap_genes_between_seq1_and_seq2()
            self._append_new_seqs_to_population()

        return self._population

    @staticmethod
    def _append_seq1_to_seqs_if_seqs_len_is_odd(seqs: list[Sequence]) -> None:
        if len(seqs) % 2 != 0:
            seqs.append(seqs[0])

    def _append_backpack_mutator_id_to_seqs(self, idx: int) -> None:
        self._seq1.seq_id += f"[bcm_{idx}] "
        self._seq2.seq_id += f"[bcm_{(idx + 1)}] "

    def _swap_genes_between_seq1_and_seq2(self) -> None:
        new_seq1_arr, new_seq2_arr = self._generate_new_seq1_and_seq2_arr_with_swapped_genes()
        self._assign_new_genes_arr_to_seq1_and_seq2(new_seq1_arr, new_seq2_arr)

    def _generate_new_seq1_and_seq2_arr_with_swapped_genes(self) -> tuple[ndarray, ndarray]:
        cross_point: int = np.random.randint(1, len(self._seq1.get_indexes_of_genes()) - 1)
        first_arr_seq_a_cp, second_arr_seq_a_cp = self._get_seq_cross_points(self._seq1, cross_point)
        first_arr_seq_b_cp, second_arr_seq_b_cp = self._get_seq_cross_points(self._seq2, cross_point)

        new_first_arr: ndarray = np.concatenate(
            (self._seq1.genes_as_arr[:first_arr_seq_a_cp],
             self._seq2.genes_as_arr[first_arr_seq_b_cp:])
        )
        new_second_arr: ndarray = np.concatenate(
            (self._seq2.genes_as_arr[:second_arr_seq_b_cp],
             self._seq1.genes_as_arr[second_arr_seq_a_cp:])
        )

        return new_first_arr, new_second_arr

    @staticmethod
    def _get_seq_cross_points(seq: Sequence, cross_point: int) -> tuple[int, int]:
        arr_first_cp: int = int(seq.get_indexes_of_genes()[cross_point])
        arr_second_cp: int = int(seq.get_indexes_of_genes()[-cross_point])
        return arr_first_cp, arr_second_cp

    def _assign_new_genes_arr_to_seq1_and_seq2(self, new_seq1_arr: ndarray,
                                               new_seq2_arr: ndarray) -> None:
        self._seq1.genes_as_arr = new_seq1_arr
        self._seq2.genes_as_arr = new_seq2_arr

    def _append_new_seqs_to_population(self) -> None:
        self._population.append(self._seq1)
        self._population.append(self._seq2)
