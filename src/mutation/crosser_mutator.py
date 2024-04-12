from dataclasses import dataclass
from typing import ClassVar

import numpy as np
from numpy import ndarray

from utils.seq import Sequence


@dataclass(init=False, repr=False)
class CrosserMutator:
    _seqs_mutated: ClassVar[list[Sequence]]
    _seq1: ClassVar[Sequence]
    _seq2: ClassVar[Sequence]

    @classmethod
    def generate_mutated_seqs(cls, seqs: list[Sequence]) -> list[Sequence]:
        cls._seqs_mutated: list[Sequence] = []
        is_seqs_len_odd: bool = (len(seqs) % 2) != 0

        seqs.append(seqs[0]) if is_seqs_len_odd else None
        cls._mutate_seqs(seqs)
        seqs.pop() if is_seqs_len_odd else None
        cls._seqs_mutated.pop() if is_seqs_len_odd else None

        return cls._seqs_mutated

    @classmethod
    def _mutate_seqs(cls, seqs: list[Sequence]) -> None:
        num_zeros: int = len(str(len(seqs)))
        for i in range(1, len(seqs), 2):
            cls._seq1, cls._seq2 = seqs[i - 1].__copy__(), seqs[i].__copy__()
            cls._append_backpack_mutator_id_to_current_seqs_id(i, num_zeros)
            cls._mutate_seq1_and_seq2(i)

    @classmethod
    def _append_backpack_mutator_id_to_current_seqs_id(cls, idx: int,
                                                       num_zeros: int) -> None:
        cls._seq1.seq_id += f"[cm_{idx:0{num_zeros}d}] "
        cls._seq2.seq_id += f"[cm_{(idx + 1):0{num_zeros}d}] "

    @classmethod
    def _mutate_seq1_and_seq2(cls, idx: int) -> None:
        cls._swap_genes_between_current_seqs()
        cls._append_current_seqs_to_seqs_mutated_list()

    @classmethod
    def _swap_genes_between_current_seqs(cls) -> None:
        new_seq1_arr, new_seq2_arr = \
            cls._generate_new_seq1_and_seq2_arr_with_swapped_genes()
        cls._seq1.genes_as_arr = new_seq1_arr
        cls._seq2.genes_as_arr = new_seq2_arr

    @classmethod
    def _generate_new_seq1_and_seq2_arr_with_swapped_genes(cls) -> tuple[ndarray, ndarray]:
        cross_point: int = np.random.randint(1, len(cls._seq1.get_indexes_of_genes()) - 1)
        seq1_arr1_cp, seq1_arr2_cp = cls._get_seq_cross_points(cls._seq1, cross_point)
        seq2_arr1_cp, seq2_arr2_cp = cls._get_seq_cross_points(cls._seq2, cross_point)

        new_seq1_arr: ndarray = np.concatenate(
            (cls._seq1.genes_as_arr[:seq1_arr1_cp],
             cls._seq2.genes_as_arr[seq2_arr1_cp:])
        )
        new_seq2_arr: ndarray = np.concatenate(
            (cls._seq2.genes_as_arr[:seq2_arr2_cp],
             cls._seq1.genes_as_arr[seq1_arr2_cp:])
        )

        return new_seq1_arr, new_seq2_arr

    @classmethod
    def _get_seq_cross_points(cls, seq: Sequence, cross_point: int) -> tuple[int, int]:
        arr_cp1: int = int(seq.get_indexes_of_genes()[cross_point])
        arr_cp2: int = int(seq.get_indexes_of_genes()[-cross_point])
        return arr_cp1, arr_cp2

    @classmethod
    def _append_current_seqs_to_seqs_mutated_list(cls) -> None:
        cls._seqs_mutated.append(cls._seq1)
        cls._seqs_mutated.append(cls._seq2)
