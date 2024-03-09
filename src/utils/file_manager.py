import os
from dataclasses import dataclass, field
from typing import ClassVar

import numpy as np
from numpy import ndarray
from utils.sequence import Sequence


@dataclass
class SeqLoader:
    @classmethod
    def load(cls, path: str) -> Sequence:
        cls.__rise_exception_if_file_is_not_valid(path)

        with open(path, 'rt', encoding='utf8') as f:
            seq_genes = ''.join(f.readlines())

        return Sequence(seq_genes)

    @classmethod
    def __rise_exception_if_file_is_not_valid(cls, file: str) -> None:
        if not os.path.exists(file):
            raise FileNotFoundError(f"File {file} not found.")

        if not file.endswith(".txt"):
            raise TypeError(f"File {file} is not a txt file.")


@dataclass(repr=False)
class SeqsSaver:
    _seq1_genes: ClassVar[str] = field(init=False)
    _seq2_genes: ClassVar[str] = field(init=False)
    _match_genes: ClassVar[str] = field(init=False)
    _COLUMNS_LEN_PER_ROW: ClassVar[int] = field(default=60)

    @classmethod
    def save(cls, seq1: Sequence, seq2: Sequence) -> None:
        cls._format_seqs_genes(seq1, seq2)
        cls._find_matches_in_seqs()
        text_lines: list[str] = cls._format_str_to_save(seq1, seq2)
        cls._save_to_file("./sequences/genes_match.txt", text_lines)

    @classmethod
    def _format_seqs_genes(cls, seq1: Sequence, seq2: Sequence) -> None:
        seqs_len_diff = seq1.genes_len - seq2.genes_len
        space_padding: str = ' ' * abs(seqs_len_diff)

        cls._seq1_genes = (seq1.genes + space_padding) if seqs_len_diff < 0 else seq1.genes
        cls._seq2_genes = (seq2.genes + space_padding) if seqs_len_diff > 0 else seq2.genes

    @classmethod
    def _find_matches_in_seqs(cls) -> None:
        # to no match mutations seq1_genes mutations are replaced: "-" to "~"
        seq1_genes = cls._seq1_genes.replace('-', '~')
        seq1: ndarray = np.array(seq1_genes, dtype='c')
        seq2: ndarray = np.array(cls._seq2_genes, dtype='c')

        matches_mask = np.char.equal(seq1, seq2)
        cls._match_genes: str = "".join(np.where(matches_mask, '|', ' ').tolist())

    @classmethod
    def _format_str_to_save(cls, seq1: Sequence, seq2: Sequence) -> list[str]:
        match_prev: int = 0
        text_lines: list[str] = []
        largest_seq_len: int = max(seq1.genes_len, seq2.genes_len)

        for i in range(0, largest_seq_len, cls._COLUMNS_LEN_PER_ROW):
            next_columns_len: int = (i + cls._COLUMNS_LEN_PER_ROW)
            num_matches_row = cls._match_genes[i:next_columns_len].count('|')

            msg: str = cls._genes_row_msg("Query", i, cls._seq1_genes)
            msg += cls._genes_match_row_msg(i, match_prev)
            msg += cls._genes_row_msg("Sbjct", i, cls._seq2_genes)

            match_prev = match_prev + num_matches_row
            text_lines.append(msg)

        return text_lines

    @classmethod
    def _genes_row_msg(cls, suffix_str: str, idx: int, seq_genes) -> str:
        next_columns_len: int = (idx + cls._COLUMNS_LEN_PER_ROW)
        seq1_genes_row: str = seq_genes[idx:next_columns_len]
        return f'\n{suffix_str:<2} {idx:<5} {seq1_genes_row:<60} {next_columns_len:>5}\n'

    @classmethod
    def _genes_match_row_msg(cls, idx: int, match_prev: int, ) -> str:
        next_columns_len: int = (idx + cls._COLUMNS_LEN_PER_ROW)
        match_genes_row: str = cls._match_genes[idx:next_columns_len]
        num_matches_row = match_genes_row.count('|')
        return (f'{"":<5} {match_prev:<5} '
                f'{match_genes_row:<60} '
                f'{(match_prev + num_matches_row):>5}')

    @classmethod
    def _save_to_file(cls, path: str, text_lines: [str]) -> None:
        with open(path, 'wt', encoding='utf8') as f:
            f.write(''.join(text_lines))
