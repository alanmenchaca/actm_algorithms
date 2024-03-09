from dataclasses import dataclass

from utils.sequence import Sequence


class SeqException(Exception):
    def __init__(self, message: str) -> None:
        self.message = message
        super().__init__(self.message)


@dataclass
class SeqsValidator(Exception):
    @classmethod
    def validate_seqs(cls, seqs: list[Sequence]) -> None:
        if (seqs is None) or not isinstance(seqs, list) or (len(seqs) == 0):
            raise SeqException('Sequence list must be a non-empty list of Sequence objects.')

        for seq in seqs:
            SeqsValidator.validate_seq(seq)

    @classmethod
    def validate_seq(cls, seq: Sequence) -> None:
        if (seq is None) or not isinstance(seq, Sequence):
            raise SeqException('Sequence object must not be None.')
