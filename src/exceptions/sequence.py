from typing import Callable

from utils.sequence import Sequence


class SequenceError(Exception):
    def __init__(self, message) -> None:
        self.message = message

    @staticmethod
    def validate_sequence(func: Callable) -> Callable[[Sequence], None]:
        def validate(self, sequence: Sequence, *args, **kwargs):
            if (sequence is None) or not isinstance(sequence, Sequence):
                raise SequenceError(f'sequence must be an instance of Sequence'
                                    f' class, got {type(sequence)} instead.')
            return func(self, sequence, *args, **kwargs)

        return validate

    @staticmethod
    def validate_sequences(func: Callable) -> Callable[[list], None]:
        def validate(self, sequences: list[Sequence], *args, **kwargs) -> None:
            if (sequences is None) or not isinstance(sequences, list) or (len(sequences) == 0):
                raise SequenceError('sequences must be a non-empty list of Sequence'
                                    f' objects, got {sequences} instead.')
            SequenceError._validate_sequence_list_has_none_values(sequences)
            return func(self, sequences, *args, **kwargs)

        return validate

    @staticmethod
    def _validate_sequence_list_has_none_values(sequences: any) -> None:
        has_none_sequence: bool = any(sequence is None for sequence in sequences)
        if has_none_sequence:
            error_msg: str = 'sequence list must not contain None values: ' \
                             'got {0} instead'.format(sequences)
            raise SequenceError(error_msg)
