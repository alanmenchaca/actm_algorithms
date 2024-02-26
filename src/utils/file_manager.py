import os
from dataclasses import dataclass, field

import numpy as np
from numpy import ndarray
from utils.sequence import Sequence


class TxtFileReader:
    @staticmethod
    def read(file: str) -> str:
        TxtFileReader.__rise_exception_if_file_is_not_valid(file)
        file_read: str = ''

        with open(file, 'rt', encoding='utf8') as f:
            for line_read in f.read():
                file_read += line_read
        return file_read

    @staticmethod
    def __rise_exception_if_file_is_not_valid(file: str) -> None:
        if not os.path.exists(file):
            raise FileNotFoundError(f"File {file} not found.")

        if not file.endswith(".txt"):
            raise TypeError(f"File {file} is not a txt file.")

    @staticmethod
    def read_files_from_path(path: str) -> [str]:
        TxtFileReader.__raise_exception_if_file_path_is_not_valid(path)

        files_read: [str] = []
        for file in os.listdir(path):
            if file.endswith(".txt"):
                files_read.append(TxtFileReader.read(f"{path}/{file}"))

        return files_read

    @staticmethod
    def __raise_exception_if_file_path_is_not_valid(path: str) -> None:
        if not os.path.exists(path):
            raise FileNotFoundError(f"File {path} not found.")

        if not os.path.isdir(path):
            raise TypeError(f"Path {path} is not a directory.")


@dataclass(init=False, repr=False)
class SequencesGenesFormatter:
    _genes_columns_len_per_row: int = field(default=60)
    _genes_to_format: ndarray = field(default=ndarray)

    _seqA_genes_formatted: ndarray = field(default=ndarray)
    _seqB_genes_formatted: ndarray = field(default=ndarray)

    def format_genes_arr(self, seqA_genes: ndarray, seqB_genes: ndarray) \
            -> tuple[ndarray, ndarray]:
        self._seqA_genes_formatted = self._format_sequence_genes(seqA_genes)
        self._seqB_genes_formatted = self._format_sequence_genes(seqB_genes)
        self._fill_sequence_with_less_rows_of_genes()

        return self._seqA_genes_formatted, self._seqB_genes_formatted

    def _format_sequence_genes(self, sequence_genes: ndarray) -> ndarray:
        self._genes_to_format: ndarray = sequence_genes
        self._pad_arr_with_gaps_to_complete_genes_columns_len_per_row()
        self._reshape_genes_arr_by_columns_len_per_row()
        return self._genes_to_format

    def _pad_arr_with_gaps_to_complete_genes_columns_len_per_row(self) -> None:
        remainder: int = self._genes_to_format.size % self._genes_columns_len_per_row
        if remainder != 0:
            padding_length: int = self._genes_columns_len_per_row - remainder
            gaps_padding_arr: ndarray = np.array(padding_length * ['-'])
            self._genes_to_format = np.append(self._genes_to_format, gaps_padding_arr)

    def _reshape_genes_arr_by_columns_len_per_row(self) -> None:
        arr_reshaped_by_columns_len_per_row: ndarray = \
            self._genes_to_format.reshape(-1, self._genes_columns_len_per_row)

        genes_formatted: list[str] = []
        for row_of_genes_as_char_arr in arr_reshaped_by_columns_len_per_row:
            row_of_genes_as_str: str = ''.join(row_of_genes_as_char_arr.tolist())
            genes_formatted.append(row_of_genes_as_str)

        self._genes_to_format = np.array(genes_formatted)

    def _fill_sequence_with_less_rows_of_genes(self) -> None:
        if len(self._seqA_genes_formatted) < len(self._seqB_genes_formatted):
            remainder: int = len(self._seqB_genes_formatted) - len(self._seqA_genes_formatted)
            self._seqA_genes_formatted = self._fill_sequence(self._seqA_genes_formatted, remainder)
        else:
            remainder: int = len(self._seqA_genes_formatted) - len(self._seqB_genes_formatted)
            self._seqB_genes_formatted = self._fill_sequence(self._seqB_genes_formatted, remainder)

    def _fill_sequence(self, seq_with_less_rows: ndarray, remainder: int) -> ndarray:
        fill_arr: list[str] = ['~' * self._genes_columns_len_per_row]

        for idx in range(remainder):
            seq_with_less_rows = np.append(seq_with_less_rows, fill_arr)

        return seq_with_less_rows


@dataclass(repr=False)
class SequencesGenesMatcher:
    _genes_matches: list[ndarray] = field(default_factory=list)

    def generate_list_of_matches_in_sequences(self, seqA_formatted_genes: ndarray,
                                              seqB_formatted_genes: ndarray) -> list[ndarray]:
        for seqA_genes, seqB_genes in zip(seqA_formatted_genes, seqB_formatted_genes):
            self._compare_genes(seqA_genes, seqB_genes)

        return self._genes_matches

    def _compare_genes(self, seqA_genes: ndarray, seqB_genes: ndarray) -> None:
        seqA_genes: ndarray = np.char.array(list(seqA_genes))
        seqB_genes: ndarray = np.char.array(list(seqB_genes))

        comparison_result: ndarray = np.char.equal(seqA_genes, seqB_genes)
        result: ndarray = np.where(comparison_result, '|', ' ')

        self._genes_matches.append(result)


@dataclass(init=False, repr=False)
class SequencesGenesMatchPrinter:
    _seqA: Sequence = field(default=Sequence)
    _seqB: Sequence = field(default=Sequence)
    matches: list[ndarray] = field(default_factory=list)

    _genes_columns_len_per_row: int = field(default=60)
    _accumulate_genes_columns_len: int = field(default=0)
    _accumulate_genes_columns_match: int = field(default=0)

    def genes_to_print(self, seqA: ndarray, seqB: ndarray, matches: list[ndarray]) -> None:
        msg: str = ""
        for seqA_row, seqB_row, match in zip(seqA, seqB, matches):
            query: str = self._get_formatted_line("Query", seqA_row)
            match_str: str = self._get_formatted_match_line(match)
            subject: str = self._get_formatted_line("Sbjct", seqB_row)

            self._accumulate_genes_columns_len += self._genes_columns_len_per_row
            msg += f'{query}\n{match_str}\n{subject}\n\n'

        print(msg)

    def _get_formatted_line(self, param: str, seq_row: ndarray | list[ndarray]) -> str:
        suffix_acc = self._accumulate_genes_columns_len
        prefix_acc = self._accumulate_genes_columns_len + self._genes_columns_len_per_row
        return f'{param:<6} {suffix_acc:<5} {seq_row} {prefix_acc:>5}'

    def _get_formatted_match_line(self, match: ndarray) -> str:
        match_join: str = ''.join(match)
        match_count: int = int(np.sum(match == '|'))
        self._accumulate_genes_columns_match += match_count
        return f"{match_join:>73} {self._accumulate_genes_columns_match:>5}"


if __name__ == '__main__':
    # seqA: Sequence = Sequence("AAAAAAAAAAAAABBBBBBBBBBBBBBBBB"
    #                           "BBBBBBBAAAAAAAAAAAAABBBBBBBBBB"
    #                           "AAAAAAAAAABBBBBBBBBBAAAAAAAAAA")
    #
    # seqB: Sequence = Sequence("AAAAAAAAAAAAABBBBBBBBBBBBBBBBB"
    #                           "BBBBBBBAAAAAAAAAAAAABAAA")

    seqA: str = TxtFileReader.read('../genes/env_HIV1H.txt')
    seqB: str = TxtFileReader.read('../genes/env_HIV1S.txt')

    seqA: Sequence = Sequence(genes=seqA)
    seqB: Sequence = Sequence(genes=seqB)

    seqA_formatted, seqB_formatted = SequencesGenesFormatter() \
        .format_genes_arr(seqA.genes_as_arr, seqB.genes_as_arr)

    matches: list[ndarray] = SequencesGenesMatcher() \
        .generate_list_of_matches_in_sequences(seqA_formatted, seqB_formatted)

    SequencesGenesMatchPrinter().genes_to_print(seqA_formatted, seqB_formatted, matches)

# @dataclass
# class TxtFileSaver:
#     @staticmethod
#     def format_and_save_sequences(seqA: Sequence, seqB: Sequence, file: str):
#         # seqA_name, seqB_name = "Sequence-A -> ", "Sequence-B -> "
#         genes_num_len: int = max(len(str(seqA.genes_len)), len(str(seqB.genes_len)))
#
#         seqA_genes_list: list[str] = TxtFileSaver._format_gene_list(seqA.genes_arr)
#         seqB_genes_list: list[str] = TxtFileSaver._format_gene_list(seqB.genes_arr)
#
#         if len(seqA_genes_list) < len(seqB_genes_list):
#             for _ in range(len(seqB_genes_list) - len(seqA_genes_list)):
#                 seqA_genes_list.extend(['~' * 60])
#
#         if len(seqA_genes_list) > len(seqB_genes_list):
#             for _ in range(len(seqA_genes_list) - len(seqB_genes_list)):
#                 seqB_genes_list.extend(['~' * 60])
#
#         # matcher_accumulator = TxtFileSaver._get_matcher_accumulator(seqA_genes_list, seqB_genes_list)
#         formatted_lines: list[str] = TxtFileSaver \
#             ._format_sequences_lines(seqA_genes_list, seqB_genes_list)
#         TxtFileSaver._save_lines_to_file(file, formatted_lines)
#
#     @staticmethod
#     def _format_gene_list(genes_arr: ndarray) -> list[str]:
#         genes_arr: ndarray = TxtFileSaver._pad_genes_array(genes_arr)
#         genes_arr: list[str] = TxtFileSaver._reshape_genes_array(genes_arr)
#         genes_arr: list[str] = TxtFileSaver._remove_zeros_from_genes_array(genes_arr)
#         return genes_arr
#
#     @staticmethod
#     def _pad_genes_array(genes_arr: ndarray) -> ndarray:
#         arr_columns_len = 60
#         if (genes_arr.size % arr_columns_len) != 0:
#             genes_padding = (arr_columns_len - genes_arr.size % arr_columns_len)
#             genes_arr = np.append(genes_arr, np.zeros(genes_padding))
#         return genes_arr
#
#     @staticmethod
#     def _reshape_genes_array(genes_arr: ndarray) -> list[str]:
#         arr_columns_len = 60
#         genes_arr = genes_arr.reshape(-1, arr_columns_len)
#         return [''.join(row_arr.tolist()) for row_arr in genes_arr]
#
#     @staticmethod
#     def _remove_zeros_from_genes_array(genes_arr: list[str]) -> list[str]:
#         return [arr_row.replace('0.0', '~') for arr_row in genes_arr]
#
#     @staticmethod
#     def _format_sequences_lines(seqA_genes_list: list[str], seqB_genes_list: list[str]) -> list[str]:
#         lines_to_save: list[str] = []    # print(f'seqB_formatted: \n{seqB_formatted}')

#         genes_seqA_acc_len: list[tuple[int, int]] = TxtFileSaver._accumulate_genes_len_pairs(seqA_genes_list)
#         genes_seqB_acc_len: list[tuple[int, int]] = TxtFileSaver._accumulate_genes_len_pairs(seqB_genes_list)
#         match_acc_pairs: list[tuple[int, int]] = TxtFileSaver._get_matcher_accumulator_pairs(seqA_genes_list,
#                                                                                              seqB_genes_list)
#         for idx, genes_row in enumerate(zip(seqA_genes_list, seqB_genes_list)):
#             seqA_row, seqB_row = genes_row
#             seqA_genes_len, seqB_genes_len = genes_seqA_acc_len[idx], genes_seqB_acc_len[idx]
#             match_acc_pair: tuple[int, int] = match_acc_pairs[idx]
#
#             seqA_str: str = TxtFileSaver._format_line("Query ", seqA_rowchar.equal, seqA_genes_len)
#             seqB_str: str = TxtFileSaver._format_line("Sbjct ", seqB_row, seqB_genes_len)
#             matcher_str: str = TxtFileSaver._format_matcher_line(seqA_row, seqB_row, match_acc_pair)
#
#             lines_to_save.extend([seqA_str, matcher_str, seqB_str, ''])
#
#         return lines_to_save
#
#     @staticmethod
#     def _accumulate_genes_len_pairs(seq_genes_list: list[str]) -> list[tuple[int, int]]:
#         accumulated_genes_len: list[tuple[int, int]] = []
#         total_length = 0
#
#         for seq_row in seq_genes_list:
#             gene_length: int = len(seq_row)
#             accumulated_genes_len.append((total_length, total_length + gene_length))
#             total_length += gene_length
#
#         return accumulated_genes_len
#
#     @staticmethod
#     def _get_matcher_accumulator_pairs(seqA_genes_list, seqB_genes_list) -> list[tuple[int, int]]:
#         accumulated_matches: list[tuple[int, int]] = []
#         suffix_matches = 0
#
#         for seqA_genes, seqB_genes in zip(seqA_genes_list, seqB_genes_list):
#             matches = np.sum(np.array(list(seqA_genes)) == np.array(list(seqB_genes)))
#             empty_spaces = np.sum(np.logical_and(np.array(list(seqA_genes)) == '~',
#                                                  np.array(list(seqB_genes)) == '~'))
#             prefix_matches = suffix_matches + matches - empty_spaces
#             accumulated_matches.append((suffix_matches, prefix_matches))
#             suffix_matches += matches
#
#         return accumulated_matches
#
#     @staticmethod
#     def _format_line(name: str, row: str, seq_genes_len: tuple[int, int]) -> str:
#         prev_acc, post_acc = seq_genes_len
#         return f"{name} {prev_acc:<5} {row} {post_acc:>5}"
#
#     @staticmethod
#     def _format_matcher_line(seqA_row: str, seqB_row: str, match_acc_pair: tuple[int, int]) -> str:
#         prev_acc, post_acc = match_acc_pair
#         matcher: ndarray = np.array(list(seqA_row)) == np.array(list(seqB_row))
#         empty_spaces_matcher: ndarray = np.logical_or(np.array(list(seqA_row)) == np.array(list("~")),
#                                                       np.array(list(seqB_row)) == np.array(list("~")))
#         matcher = matcher & ~empty_spaces_matcher
#         matcher: ndarray = np.where(matcher, "|", " ")
#         return f"\t   {prev_acc:<5} {''.join(matcher):>58} {post_acc:>5}"
#
#     @staticmethod
#     def _save_lines_to_file(file: str, lines: list[str]) -> None:
#         with open(file, 'wt', encoding='utf8') as f:
#             f.write(f'\t\t[Best Sequences file]\n')
#             f.write('\t\tMétodos implementados: Simple Mutator + Backpack Crosser Mutator\n\n')
#             f.write('\t\t* Poblaciones totales: 10\n')
#             f.write('\t\t* Secuencias por población: 100\n')
#             f.write('\t\t* Secuencias Totales: 2,000\n\n')
#             f.write('\t\t\t- seqA: Sequence-A\n')
#             f.write('\t\t\t- seqB: Sequence-B\n\n')
#
#         lines = [f'\t{line}\n' for line in lines]
#         with open(file, 'at', encoding='utf8') as f:
#             f.write(''.join(lines))
