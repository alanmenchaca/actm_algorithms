from dataclasses import dataclass, field

import numpy as np
from numpy import ndarray

from exceptions.sequence import SeqsValidator as sv
from utils.sequence import Sequence


@dataclass(repr=False)
class SimpleMutator:
    _seq: Sequence = field(init=True, default=Sequence)
    _rand_indexes_len: int = field(default=None)
    _rand_indexes_arr: ndarray = field(default=None)
    _gaps_lengths_arr: ndarray = field(default=None)

    _is_rand_indexes_len_set_manually: bool = field(default=False)
    _is_gaps_lengths_arr_set_manually: bool = field(default=False)

    @property
    def rand_indexes_len(self) -> int:
        return self._rand_indexes_len

    @rand_indexes_len.setter
    def rand_indexes_len(self, rand_indexes_range: tuple[int, int]) -> None:
        self._check_low_high_values(rand_indexes_range)
        low, high = rand_indexes_range
        self._rand_indexes_len = np.random.randint(low, high + 1)
        self._is_rand_indexes_len_set_manually = True

    @property
    def gaps_lengths_arr(self) -> ndarray:
        return self._gaps_lengths_arr

    @gaps_lengths_arr.setter
    def gaps_lengths_arr(self, gaps_lengths_range: tuple[int, int]) -> None:
        self._check_low_high_values(gaps_lengths_range)
        low, high = gaps_lengths_range
        self._gaps_lengths_arr = np.random.randint(low, high + 1, self._rand_indexes_len)
        self._is_gaps_lengths_arr_set_manually = True
        self._check_gaps_lengths_arr_is_not_int()

    def _check_gaps_lengths_arr_is_not_int(self) -> None:
        if isinstance(self._gaps_lengths_arr, int):
            self._gaps_lengths_arr = np.array([self._gaps_lengths_arr])

    @staticmethod
    def _check_low_high_values(low_high_range: tuple[int, int]) -> None:
        low, high = low_high_range
        if not (0 <= low < high):
            raise ValueError("The range must be 0 <= low < high, "
                             f"the range obtained: '{low_high_range}' is invalid.")

    def generate_mutated_population(self, seq: Sequence, num_seqs: int) -> list[Sequence]:
        sv.validate_seq(seq)
        self._validate_num_seqs(num_seqs)

        mutated_population: list[Sequence] = [seq.__copy__() for _ in range(num_seqs)]
        self.mutate_seqs_genes(mutated_population)

        return mutated_population

    @staticmethod
    def _validate_num_seqs(num_seqs: int) -> None:
        if (num_seqs is None) or (num_seqs < 1):
            raise ValueError("The number of sequences must be grater than 0,"
                             f" but got {num_seqs}.")

    def mutate_seqs_genes(self, seqs: list[Sequence]) -> None:
        sv.validate_seqs(seqs)
        for idx, seq in enumerate(seqs):
            seq.seq_id += f"[sm_{(idx + 1)}] "
            self._mutate_genes(seq)

    def _mutate_genes(self, seq: Sequence) -> None:
        sv.validate_seq(seq)
        self._init_mutator_params_when_mutate_genes(seq)
        self._rand_indexes_arr, self._gaps_lengths_arr = zip(*sorted(
            zip(self._rand_indexes_arr, self._gaps_lengths_arr), reverse=True))
        self._insert_gaps_to_seq()

    def _init_mutator_params_when_mutate_genes(self, seq: Sequence) -> None:
        self._seq = seq
        self._init_rand_indexes_len_if_is_none()
        self._init_gaps_lengths_arr_if_is_none()
        self._rand_indexes_arr: ndarray = np.random.choice(seq.genes_len + 1,
                                                           self._rand_indexes_len, replace=False)

    def _init_rand_indexes_len_if_is_none(self) -> None:
        if not self._is_rand_indexes_len_set_manually:
            # 1 <= rand_indexes_len <= 6
            self._rand_indexes_len: int = np.random.randint(1, 7)

    def _init_gaps_lengths_arr_if_is_none(self) -> None:
        if not self._is_gaps_lengths_arr_set_manually:
            # 1 <= gaps_lengths_arr <= 3
            self._gaps_lengths_arr: ndarray = np.random.randint(1, 4, self._rand_indexes_len)

    def _insert_gaps_to_seq(self) -> None:
        for idx, num_gaps in zip(self._rand_indexes_arr, self._gaps_lengths_arr):
            gap_arr_to_insert: ndarray = np.array(['-'] * num_gaps)
            self._seq.genes_as_arr = np.insert(self._seq.genes_as_arr, idx, gap_arr_to_insert)

    def remove_mutations_randomly(self, seqs: list[Sequence]) -> None:
        sv.validate_seqs(seqs)
        for seq in seqs:
            self._remove_gaps_to_seq(seq)

    def _remove_gaps_to_seq(self, seq: Sequence) -> None:
        self._init_mutator_params_when_remove_mutations(seq)
        seq.genes_as_arr = np.delete(seq.genes_as_arr, self._rand_indexes_arr)

    def _init_mutator_params_when_remove_mutations(self, seq: Sequence) -> None:
        gap_indices: ndarray = np.where(seq.genes_as_arr == '-')[0]
        self._rand_indexes_len: int = np.random.randint(0, 6)
        self._rand_indexes_arr: ndarray = np.random.choice(gap_indices, self._rand_indexes_len)
