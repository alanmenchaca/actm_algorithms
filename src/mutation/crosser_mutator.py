import copy
from dataclasses import dataclass
from typing import ClassVar

import numpy as np

from utils.seq import Sequence


@dataclass(init=False, repr=False)
class CrosserMutator:
    _seqs_mutated: ClassVar[list[Sequence]]
    _seq1: ClassVar[Sequence]
    _seq2: ClassVar[Sequence]

    @classmethod
    def generate_mutated_seqs(cls, seqs: list[Sequence]) -> list[Sequence]:
        cls._seqs_mutated: list[Sequence] = copy.deepcopy(seqs)
        is_seqs_len_odd: bool = (len(seqs) % 2) != 0

        cls._seqs_mutated.append(seqs[0]) if is_seqs_len_odd else None
        cls._mutate_seqs()
        cls._seqs_mutated.pop() if is_seqs_len_odd else None

        return cls._seqs_mutated

    @classmethod
    def _mutate_seqs(cls) -> None:
        num_zeros: int = len(str(len(cls._seqs_mutated)))
        for i in range(1, len(cls._seqs_mutated), 2):
            cls._seq1, cls._seq2 = cls._seqs_mutated[i - 1], cls._seqs_mutated[i]
            cls._append_backpack_mutator_id_to_current_seqs_id(i, num_zeros)
            cls._swap_genes_between_current_seqs()

    @classmethod
    def _append_backpack_mutator_id_to_current_seqs_id(cls, idx: int,
                                                       num_zeros: int) -> None:
        cls._seq1.seq_id += f"[cm_{idx:0{num_zeros}d}] "
        cls._seq2.seq_id += f"[cm_{(idx + 1):0{num_zeros}d}] "

    @classmethod
    def _swap_genes_between_current_seqs(cls) -> None:
        cross_point: int = np.random.randint(1, len(cls._seq1.get_indexes_of_genes()) - 1)
        seq1_arr1_cp, seq1_arr2_cp = cls._get_seq_cross_points(cls._seq1, cross_point)
        seq2_arr1_cp, seq2_arr2_cp = cls._get_seq_cross_points(cls._seq2, cross_point)

        cls._seq1.genes_as_arr = np.concatenate(
            (cls._seq1.genes_as_arr[:seq1_arr1_cp],
             cls._seq2.genes_as_arr[seq2_arr1_cp:])
        )
        cls._seq2.genes_as_arr = np.concatenate(
            (cls._seq2.genes_as_arr[:seq2_arr2_cp],
             cls._seq1.genes_as_arr[seq1_arr2_cp:])
        )

    @classmethod
    def _get_seq_cross_points(cls, seq: Sequence, cross_point: int) -> tuple[int, int]:
        return (int(seq.get_indexes_of_genes()[cross_point]),
                int(seq.get_indexes_of_genes()[-cross_point]))
