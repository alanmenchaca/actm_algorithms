import numpy as np
from numpy import ndarray


class GenesError(Exception):
    def __init__(self, message) -> None:
        self.message = message

    @staticmethod
    def validate_genes(genes: any) -> None:
        if (genes is None) or not isinstance(genes, str) or (len(genes) == 0):
            error_msg: str = 'genes must be a non-empty string, got {0} instead'.format(type(genes))
            raise GenesError(error_msg)
        if not genes.isalpha() and not ('-' in genes):
            raise GenesError('Genes string must contains only letters or gaps (mutations).')

    @staticmethod
    def validate_genes_as_arr(genes_as_arr: any) -> None:
        if (genes_as_arr is None) or not isinstance(genes_as_arr, ndarray) or (genes_as_arr.size == 0):
            error_msg: str = 'genes_arr must be a non-empty 2D array, got {0} instead'.format(type(genes_as_arr))
            raise GenesError(error_msg)
        if genes_as_arr.dtype != np.dtype('U1'):
            raise GenesError('Genes array must contains only strings of length 1 (a char).')
        if not np.any(np.char.isalpha(genes_as_arr)) and not np.any(genes_as_arr == '-'):
            raise GenesError('Genes array must contains only letters or gaps (mutation).')
