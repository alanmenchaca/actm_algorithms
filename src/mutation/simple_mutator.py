from dataclasses import dataclass
from typing import ClassVar

import numpy as np
from numpy import ndarray

from utils.seq import Sequence


@dataclass
class SimpleMutator:
    _rand_indexes_len: ClassVar[int] = None
    _gaps_lengths_arr: ClassVar[ndarray] = None
    _rand_indexes_arr: ClassVar[ndarray] = None

    _rand_indexes_len_range: ClassVar[tuple[int, int]] = None
    _gaps_lengths_arr_range: ClassVar[tuple[int, int]] = None

    @classmethod
    def set_params(cls, rand_indexes_len_range: tuple[int, int],
                   gaps_lengths_arr_range: tuple[int, int]) -> None:
        cls._rand_indexes_len_range = rand_indexes_len_range
        cls._gaps_lengths_arr_range = gaps_lengths_arr_range

    @classmethod
    def generate_mutated_seqs(cls, seq: Sequence, num_seqs: int) -> list[Sequence]:
        seqs_mutated: list[Sequence] = [seq.__copy__() for _ in range(num_seqs)]
        cls.mutate_seqs_genes(seqs_mutated)
        return seqs_mutated

    @classmethod
    def mutate_seqs_genes(cls, seqs: list[Sequence]) -> None:
        num_zeros: int = len(str(len(seqs)))
        for i, seq in enumerate(seqs):
            seq.seq_id += f"[sm_{(i + 1):0{num_zeros}d}] "
            cls._mutate_genes(seq)

    @classmethod
    def _mutate_genes(cls, seq: Sequence) -> None:
        cls._init_rand_indexes_len()
        cls._init_gaps_lengths_arr()
        cls._init_rand_indexes_arr(seq)
        cls._insert_gaps_to_seq(seq)

    @classmethod
    def _init_rand_indexes_len(cls) -> None:
        # default range: [3, 6]
        low, high = cls._rand_indexes_len_range if cls._rand_indexes_len_range else (3, 6)
        cls._rand_indexes_len = np.random.randint(low, high + 1)

    @classmethod
    def _init_gaps_lengths_arr(cls) -> None:
        # default range: [3, 6]
        low, high = cls._gaps_lengths_arr_range if cls._gaps_lengths_arr_range else (3, 7)
        cls._gaps_lengths_arr = np.random.randint(low, high + 1, cls._rand_indexes_len)

    @classmethod
    def _init_rand_indexes_arr(cls, seq: Sequence) -> None:
        cls._rand_indexes_arr = \
            np.random.choice(seq.genes_len + 1, cls._rand_indexes_len, replace=False)
        cls._rand_indexes_arr, cls._gaps_lengths_arr = \
            zip(*sorted(zip(cls._rand_indexes_arr, cls._gaps_lengths_arr), reverse=True))

    @classmethod
    def _insert_gaps_to_seq(cls, seq: Sequence) -> None:
        for idx, num_gaps in zip(cls._rand_indexes_arr, cls._gaps_lengths_arr):
            gap_arr_to_insert: ndarray = np.array(['-'] * num_gaps)
            seq.genes_as_arr = np.insert(seq.genes_as_arr, idx, gap_arr_to_insert)
