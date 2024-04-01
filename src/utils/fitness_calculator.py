from dataclasses import dataclass, field
from typing import ClassVar

import numpy as np

from utils.sequence import Sequence


@dataclass(repr=False)
class FitnessCalculator:
    _main_seq: ClassVar[Sequence] = field(default=None)
    _current_seq: ClassVar[Sequence] = field(init=False, default=None)

    @classmethod
    def compute_seqs_fitness(cls, main_seq: Sequence, seqs: list[Sequence]) -> None:
        for seq in seqs:
            cls._main_seq = main_seq.__copy__()
            cls._compute_seq_fitness(seq)

        seqs.sort(key=lambda seq: seq.fitness, reverse=True)

    @classmethod
    def _compute_seq_fitness(cls, seq: Sequence) -> None:
        cls._current_seq = seq.__copy__()
        cls._match_genes_between_main_and_current_seq()
        seq.fitness = int(cls._count_genes_match())

    @classmethod
    def _match_genes_between_main_and_current_seq(cls) -> None:
        abs_genes_diff: int = cls._main_seq.genes_len - cls._current_seq.genes_len
        # main_seq genes are bigger if abs_genes_diff > 0
        if abs_genes_diff > 0:
            cls._append_gaps_to_lower_seqs_genes(cls._main_seq, cls._current_seq)
        elif abs_genes_diff < 0:
            cls._append_gaps_to_lower_seqs_genes(cls._current_seq, cls._main_seq)

    @classmethod
    def _append_gaps_to_lower_seqs_genes(cls, bigger_seq: Sequence, lower_seq: Sequence) -> None:
        genes_length_difference: int = (bigger_seq.genes_len - lower_seq.genes_len)
        lower_seq.genes += '-' * genes_length_difference

    @classmethod
    def _count_genes_match(cls) -> int:
        cls._main_seq.genes = cls._main_seq.genes.replace('-', '~')
        matches_mask = np.char.equal(cls._main_seq.genes_as_arr, cls._current_seq.genes_as_arr)
        return np.count_nonzero(matches_mask)
