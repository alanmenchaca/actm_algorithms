from dataclasses import dataclass

import numpy as np
from numpy import ndarray


class GenesException(Exception):
    def __init__(self, message: str) -> None:
        self.message = message
        super().__init__(self.message)


@dataclass
class GenesValidator:
    @classmethod
    def validate_genes_str(cls, genes: any) -> None:
        if not isinstance(genes, str):
            raise GenesException('Genes must be a string type, got {0} instead.'.format(type(genes)))

        if len(genes) == 0:
            raise GenesException('Sequence genes must be a non-empty string.')

        if not genes.isalpha() and not ('-' in genes):
            raise GenesException('Genes string must contains only letters or gaps (mutations).')

    @classmethod
    def validate_genes_as_arr(cls, genes_as_arr: ndarray) -> None:
        if not isinstance(genes_as_arr, ndarray):
            raise GenesException('Genes must be a numpy array type, got {0} instead.'
                                 .format(type(genes_as_arr)))

        if genes_as_arr.size == 0:
            raise GenesException('Genes array must be a non-empty 2D numpy array')

        if genes_as_arr.dtype != 'U1':
            raise GenesException('Genes array must contains only strings of length 1 (a char).')

        if not np.any(np.char.isalpha(genes_as_arr)) and not np.any(genes_as_arr == '-'):
            raise GenesException('Genes array must contains only letters or gaps (mutation).')
