import copy
from dataclasses import dataclass, field

import numpy as np
from numpy import ndarray

from exceptions.sequence import SeqsValidator as sv
from utils.sequence import Sequence


@dataclass(repr=False)
class FitnessCalculator:
    _main_seq: Sequence = field(default=None)
    _main_seq_temp: Sequence = field(init=False, default=None)
    _secondary_seq: Sequence = field(init=False, default=None)

    _best_seq: Sequence = field(init=False, default=None)
    _best_seqs: dict[str, Sequence] = field(init=False)

    def set_main_seq(self, seq: Sequence) -> None:
        sv.validate_seq(seq)
        self._main_seq = seq
        self._main_seq_temp = copy.copy(seq)

        if self._best_seq is None:
            self._best_seq = copy.copy(seq)

    def set_secondary_seq(self, seq: Sequence) -> None:
        sv.validate_seq(seq)
        self._secondary_seq = copy.copy(seq)

    def compute_fitness(self, seqs: list[Sequence]) -> None:
        sv.validate_seqs(seqs)
        self._check_if_main_seq_is_set()
        seqs_fitness: list[float] = []

        for seq in seqs:
            self.set_secondary_seq(seq)
            seq.fitness = self.compute_fitness_between_main_and_secondary_seqs()
            seqs_fitness.append(seq.fitness)

        best_seq_idx: ndarray = np.argmax(seqs_fitness)
        current_best_seq: Sequence = seqs[best_seq_idx]
        self._set_best_seqs(current_best_seq)

    def _check_if_main_seq_is_set(self) -> None:
        if self._main_seq is None:
            raise ValueError('You must set main seq before computing fitness.')

    def compute_fitness_between_main_and_secondary_seqs(self) -> float:
        if self._main_seq_temp.genes_len != self._secondary_seq.genes_len:
            self._match_genes_between_main_and_secondary_seqs_genes()

        return self._count_genes_match()

    def _match_genes_between_main_and_secondary_seqs_genes(self) -> None:
        # by default, the main seq is the one with the biggest sequences length
        bigger_seq: Sequence = self._main_seq_temp
        lower_seq: Sequence = self._secondary_seq

        # if the secondary seq has the biggest sequences length, then we swap the seqs
        if self._main_seq_temp.genes_len < self._secondary_seq.genes_len:
            bigger_seq, lower_seq = self._secondary_seq, self._main_seq_temp

        self._append_gaps_to_lower_seqs_genes(lower_seq, bigger_seq)

    @staticmethod
    def _append_gaps_to_lower_seqs_genes(lower_seq: Sequence, bigger_seq: Sequence) -> None:
        genes_length_difference: int = (bigger_seq.genes_len - lower_seq.genes_len)
        gaps_arr: ndarray = np.array(['-'] * genes_length_difference)
        lower_seq.genes_as_arr = np.append(lower_seq.genes_as_arr, gaps_arr)

    def _count_genes_match(self) -> int:
        genes_match_mask: ndarray = np.equal(self._main_seq_temp.genes_as_arr, self._secondary_seq.genes_as_arr)
        gaps_mask: ndarray = np.logical_or(self._secondary_seq.genes_as_arr != '-',
                                           self._main_seq_temp.genes_as_arr != '-')
        genes_without_mutations_mask: ndarray = np.logical_and(genes_match_mask, gaps_mask)
        return np.count_nonzero(genes_without_mutations_mask)

    def _set_best_seqs(self, current_seq: Sequence) -> None:
        if self._best_seq.fitness <= current_seq.fitness:
            self._main_seq.fitness = current_seq.fitness
            self._best_seq = current_seq
            self._best_seqs = {'main_seq': self._main_seq,
                               'best_seq': current_seq}

    def get_best_seqs(self) -> dict[str, Sequence]:
        return self._best_seqs
