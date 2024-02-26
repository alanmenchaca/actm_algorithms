import copy
from dataclasses import dataclass, field

import numpy as np
from numpy import ndarray

from exceptions.sequence import SequenceError
from utils.sequence import Sequence


@dataclass(repr=False)
class FitnessCalculator:
    _seqA: Sequence = field(default=None)
    _seqA_temp: Sequence = field(init=False, default=None)
    _seqB: Sequence = field(init=False, default=None)

    _best_sequence: Sequence = field(init=False, default=None)
    _best_sequences: dict[str, Sequence] = field(init=False)

    @SequenceError.validate_sequence
    def set_seqA(self, sequence: Sequence) -> None:
        self._seqA = sequence
        self._seqA_temp = copy.copy(sequence)

        if self._best_sequence is None:
            self._best_sequence = copy.copy(sequence)

    @SequenceError.validate_sequence
    def set_seqB(self, sequence: Sequence) -> None:
        self._seqB = copy.copy(sequence)

    @SequenceError.validate_sequences
    def compute_fitness(self, sequences: list[Sequence]) -> None:
        self._check_if_seqA_is_set()
        sequences_fitness: list[float] = []

        for sequence in sequences:
            self.set_seqB(sequence)
            sequence.fitness = self.compute_fitness_between_seqA_and_seqB()
            sequences_fitness.append(sequence.fitness)

        best_sequence_idx: ndarray = np.argmax(sequences_fitness)
        current_best_sequence: Sequence = sequences[best_sequence_idx]
        self._set_best_sequences(current_best_sequence)

    def _check_if_seqA_is_set(self) -> None:
        if self._seqA is None:
            raise ValueError('You must set seqA sequence before computing fitness.')

    def compute_fitness_between_seqA_and_seqB(self) -> float:
        if self._seqA_temp.genes_length != self._seqB.genes_length:
            self._match_genes_between_seqA_and_seqB_sequences_genes()

        return self._count_genes_match()

    def _match_genes_between_seqA_and_seqB_sequences_genes(self) -> None:
        # by default, the seqA sequence is the one with the biggest genes length
        bigger_sequence: Sequence = self._seqA_temp
        lower_sequence: Sequence = self._seqB

        # if the seqB sequence has the biggest genes length, then we swap the sequences
        if self._seqA_temp.genes_length < self._seqB.genes_length:
            bigger_sequence, lower_sequence = self._seqB, self._seqA_temp

        self._append_gaps_to_lower_sequence_genes(lower_sequence, bigger_sequence)

    @staticmethod
    def _append_gaps_to_lower_sequence_genes(lower_sequence: Sequence,
                                             bigger_sequence: Sequence) -> None:
        genes_length_difference: int = (bigger_sequence.genes_length - lower_sequence.genes_length)
        gaps_arr: ndarray = np.array(['-'] * genes_length_difference)
        lower_sequence.genes_as_arr = np.append(lower_sequence.genes_as_arr, gaps_arr)

    def _count_genes_match(self) -> int:
        genes_match_mask: ndarray = np.equal(self._seqA_temp.genes_as_arr, self._seqB.genes_as_arr)
        gaps_mask: ndarray = np.logical_or(self._seqB.genes_as_arr != '-',
                                           self._seqA_temp.genes_as_arr != '-')
        genes_without_mutations_mask: ndarray = np.logical_and(genes_match_mask, gaps_mask)
        return np.count_nonzero(genes_without_mutations_mask)

    def _set_best_sequences(self, current_sequence: Sequence):
        if self._best_sequence.fitness <= current_sequence.fitness:
            self._seqA.fitness = current_sequence.fitness
            self._best_sequence = current_sequence
            self._best_sequences = {'seqA': self._seqA, 'seqB': current_sequence}

    def get_best_sequence_of_all_populations(self) -> Sequence:
        return self._best_sequence

    def get_best_sequences_of_each_populations(self) -> dict[str, Sequence]:
        return self._best_sequences
