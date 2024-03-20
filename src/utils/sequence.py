import copy
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from numpy import ndarray

from utils.molecule import Molecule


@dataclass
class Sequence:
    _genes: str = ""
    seq_id: Optional[str] = ""
    _genes_as_arr: ndarray = None
    _genes_len: int = 0

    _fitness: float = 0.0
    _molecule: Molecule = None

    def __post_init__(self) -> None:
        self._genes_len = len(self.genes)
        self._genes_as_arr = (np.array(list(self.genes), dtype='c')
                              .reshape((self._genes_len, 1)))
        self._molecule = Molecule()

    @property
    def genes(self) -> str:
        return self._genes

    @genes.setter
    def genes(self, genes: str) -> None:
        self._genes = genes
        self._genes_len = len(genes)
        self._genes_as_arr = (np.array(list(genes), dtype='c')
                              .reshape((self._genes_len, 1)))

    @property
    def genes_as_arr(self) -> ndarray:
        return self._genes_as_arr

    @genes_as_arr.setter
    def genes_as_arr(self, genes_arr: ndarray) -> None:
        self._genes_len = len(genes_arr)
        self._genes_as_arr = genes_arr.reshape((self._genes_len, 1))
        self.genes = self._genes_as_arr.tobytes().decode('utf-8')

    @property
    def fitness(self) -> float:
        return self._fitness

    @fitness.setter
    def fitness(self, fitness: float) -> None:
        self._fitness = fitness
        self._molecule.potential_energy = fitness

    @property
    def genes_len(self) -> int:
        return self._genes_len

    def get_genes_without_mutations(self) -> str:
        return self.genes.replace('-', '')

    def get_genes_arr_without_mutations(self) -> ndarray:
        genes_without_mutations_len: int = len(self.get_genes_without_mutations())
        genes_str_list: list[str] = list(self.get_genes_without_mutations())
        return np.array(genes_str_list).reshape((genes_without_mutations_len, 1))

    def was_seq_mutated(self) -> bool:
        return '-' in self.genes

    def get_indexes_of_genes(self) -> ndarray:
        return np.where(self._genes_as_arr != b'-')[0]

    def to_molecule_instance(self) -> Molecule:
        return self._molecule

    def __eq__(self, other: 'Sequence') -> bool:
        return (other.genes == self.genes
                and np.array_equal(other._genes_as_arr, self._genes_as_arr)
                and other._genes_len == self._genes_len)

    def __copy__(self) -> 'Sequence':
        return copy.deepcopy(self)

    def __repr__(self) -> str:
        return f'Sequence(genes: {self.genes[:10]}...,' \
               f' genes_len: {self._genes_len},' \
               f' fitness: {self._fitness})'
