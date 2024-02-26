from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from numpy import ndarray

from src.exceptions.genes import GenesError


@dataclass()
class Molecule:
    potential_energy: int = field(default=0, init=False)
    kinetic_energy: float = field(default=0.0, init=False)

    @property
    def total_energy(self) -> float:
        return self.potential_energy + self.kinetic_energy


@dataclass()
class Sequence:
    genes: str = ""
    seq_id: Optional[str] = ""
    _genes_as_arr: ndarray = field(default=ndarray, init=False)
    _genes_length: int = field(default=int, init=False)
    _fitness: float = field(default=0.0, init=False)

    _molecule: Molecule = field(default=Molecule, init=False)

    def __post_init__(self) -> None:
        GenesError.validate_genes(self.genes)
        self._genes_length: int = len(self.genes)
        self._genes_as_arr: ndarray = np.array(list(self.genes)) \
            .reshape((self._genes_length, 1))
        self._molecule: Molecule = Molecule()

    @property
    def genes_as_arr(self) -> ndarray:
        return self._genes_as_arr

    @genes_as_arr.setter
    def genes_as_arr(self, genes_arr: ndarray) -> None:
        GenesError.validate_genes_as_arr(genes_arr)
        self._genes_length: int = len(genes_arr)
        self._genes_as_arr: ndarray = genes_arr.reshape((self._genes_length, 1))
        self.genes: str = "".join(self._genes_as_arr.flatten())

    @property
    def fitness(self) -> float:
        return self._fitness

    @fitness.setter
    def fitness(self, fitness: float) -> None:
        self._fitness: float = fitness
        self._molecule.potential_energy = fitness

    @property
    def genes_length(self) -> int:
        return self._genes_length

    def get_genes_without_mutations(self) -> str:
        return self.genes.replace('-', '')

    def get_genes_arr_without_mutations(self) -> ndarray:
        genes_without_mutations_len = len(self.get_genes_without_mutations())
        return np.array(list(self.get_genes_without_mutations())) \
            .reshape((genes_without_mutations_len, 1))

    def was_sequence_mutated(self) -> bool:
        return '-' in self.genes

    def get_indexes_of_genes(self) -> ndarray:
        return np.where(self._genes_as_arr != '-')[0]

    def to_molecule_instance(self) -> Molecule:
        return self._molecule

    def __eq__(self, other) -> bool:
        return (other.genes == self.genes
                and np.array_equal(other._genes_as_arr, self._genes_as_arr)
                and other._genes_length == self._genes_length)

    def __copy__(self) -> 'Sequence':
        return Sequence(self.genes)

    def __repr__(self) -> str:
        msg: str = f'Sequence(genes: {self.genes[:10]}...,' \
                   f' genes_len: {self._genes_length},' \
                   f' fitness: {self._fitness}),'
        return msg
