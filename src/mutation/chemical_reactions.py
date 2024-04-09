from dataclasses import dataclass, field
from typing import Callable, ClassVar

import numpy as np

from mutation.crosser_mutator import CrosserMutator as BCMutator
from mutation.simple_mutator import SimpleMutator
from utils.fitness_calculator import FitnessCalculator
from utils.sequence import Sequence, Molecule


@dataclass(repr=False)
class ChemicalReactionsMutator:
    """
    Una molécula posee dos tipos de energías:
        * Energía Potencial (PE)
        * Energía Cinética (KE)
        - Energía total (TE = PE + KE)

    Una estructura molécula tiene la intención de cambiar de w a w'.
        * El cambio siempre es posible si PE_w >= PE_w'.

    De lo contrario, permitimos el cambio solo cuando:
        * PE_w + KE_w >= PE_w'

    Cuanto mayor sea la KE de la molécula, mayor será la posibilidad puede
    poseer una nueva estructura molecular con mayor PE.

    * Una estructura más favorable es aquella que tiene el estado potencial
     más bajo posible.
     * Las moléculas involucradas en una reacción intentan alcanzar el estado
    potencial más bajo posible, pero la búsqueda ciega de estructuras más
    favorables dará como resultado estados metaestables (estructuras menos
    favorables, es decir, atascarse en mínimos locales).

    * KE (Energía Cinética) permite que las moléculas se muevan a un estado
    de mayor potencial y, por lo tanto, la posibilidad de tener una estructura
    más favorable en un cambio futuro.
    * En consecuencia, la Energía Cinética (KE) de una molécula simboliza su
    capacidad de escapar de un mínimo local.
    """
    _seq_to_compare: ClassVar[Sequence] = None
    _seq1: ClassVar[Sequence] = None
    _seq2: ClassVar[Sequence] = None

    _buffer: ClassVar[float] = field(default=0.01)

    @classmethod
    def collide_molecules(cls, seq_to_compare: Sequence, seqs: list[Sequence],
                          num_collisions: int = 100) -> None:
        for i in range(num_collisions):
            cls._seq_to_compare = seq_to_compare

            # choose two random sequences
            seq1_idx, seq2_idx = np.random.choice(len(seqs), 2, replace=False)
            cls._seq1, cls._seq2 = seqs[seq1_idx], seqs[seq2_idx]

            cls._seq1.seq_id += f"[crm] " if "[crm]" not in cls._seq1.seq_id else ""
            cls._seq2.seq_id += f"[crm] " if "[crm]" not in cls._seq2.seq_id else ""

            # select a random collision type
            collision_type: int = np.random.randint(0, 4)
            cls._do_collision(seqs, collision_type)

        seq_to_compare.fitness = 0

    @classmethod
    def _do_collision(cls, seqs: list[Sequence], collision_type: int) -> None:
        collisions: list[Callable] = [
            cls._do_collision_type0,
            cls._do_collision_type1,
            cls._do_collision_type2,
            cls._do_collision_type3
        ]
        collisions[collision_type](seqs)

    @classmethod
    def _do_collision_type0(cls, seqs: list[Sequence]) -> None:
        _, cls._buffer = cls.ineffective_collision_against_the_wall(cls._seq_to_compare,
                                                                    cls._seq1, cls._buffer)

    @classmethod
    def _do_collision_type1(cls, seqs: list[Sequence]) -> None:
        new_seq1, new_seq2, success, cls._buffer = cls \
            .decomposition(cls._seq_to_compare, cls._seq1, cls._buffer)
        if success:
            seqs.remove(cls._seq1)
            seqs.append(new_seq1)
            seqs.append(new_seq2)

    @classmethod
    def _do_collision_type2(cls, seqs: list[Sequence]) -> None:
        _, _ = cls.ineffective_intermolecular_collision(cls._seq_to_compare, cls._seq1, cls._seq2)

    @classmethod
    def _do_collision_type3(cls, seqs: list[Sequence]) -> None:
        generated_seq, success = cls.synthesis(cls._seq_to_compare, cls._seq1, cls._seq2)
        if success:
            seqs.remove(cls._seq1)
            seqs.remove(cls._seq2)
            seqs.append(generated_seq)

    @classmethod  # Input: A molecule M with its profile and central energy buffer.
    def ineffective_collision_against_the_wall(cls, seq_to_compare: Sequence, seq: Sequence,
                                               buffer: float) -> tuple[Sequence, float]:
        """
            ~ Colisión ineficaz contra la pared ~
        Una colisión ineficaz en la pared ocurre cuando: Una molécula golpea
         la pared y luego rebota.

            * Algunos atributos moleculares cambian en esta colisión y, por lo tanto, la
            estructura molecular varía en consecuencia.
            * Como la colisión no es tan fuerte, la estructura molecular resultante no
            debería ser demasiado diferente de la original.
        """
        molecule: Molecule = seq.to_molecule_instance()

        # new sequence with a new molecular structure
        # Obtain w' = Neighbor(w)
        seq_prime: Sequence = seq.__copy__()
        SimpleMutator.mutate_seqs_genes([seq_prime])

        # Calculate PE_w'
        pe_prime: float = cls._compute_potential_energy(seq_to_compare, seq_prime)

        # Change is allowed if: PE_w + KE_w >= PE_w'
        if molecule.total_energy >= pe_prime:
            # Get q randomly in interval [KELossRate, 1]
            ke_loss_rate: float = np.random.uniform(0, 1)

            # KE_w' = (PE_w + KE_w - PE_w') * q
            ke_prime: float = (molecule.total_energy - pe_prime) * ke_loss_rate

            # buffer = buffer + (PE_w + KE_w - PE_w') * (1 - q)
            buffer: float = (buffer + molecule.total_energy - pe_prime) * (1 - ke_loss_rate)

            # Update the profile o M by w = w'. PE_w = PE_w' and KE_w = KE_w'
            seq.genes = seq_prime.genes
            seq.fitness = pe_prime
            molecule.potential_energy = pe_prime
            molecule.kinetic_energy = ke_prime

        # Output M and buffer
        return seq, buffer

    @classmethod  # Input: A Molecule M with its profile and the central energy buffer.
    def decomposition(cls, seq_to_compare: Sequence, seq: Sequence,
                      buffer: float) -> tuple[Sequence, Sequence, bool, float]:
        """
            ~ Descomposición ~
        Una descomposición significa que una molécula golpea la pared y luego
         se descompone en dos o más piezas.

            * La colisión es vigorosa y hace que la molécula se rompa en dos pedazos.
            * Las estructuras moleculares resultantes deberían ser muy diferentes de
            la original.
         """
        molecule: Molecule = seq.to_molecule_instance()

        # Create new molecules M_1' and M_2'
        seq1: Sequence = seq.__copy__()
        seq2: Sequence = seq.__copy__()

        molecule1_prime: Molecule = seq1.to_molecule_instance()
        molecule2_prime: Molecule = seq2.to_molecule_instance()

        # Obtain w_1' and w_2' from w
        seq1_prime: Sequence = seq1.__copy__()
        seq2_prime: Sequence = seq2.__copy__()

        SimpleMutator.mutate_seqs_genes([seq1_prime, seq2_prime])

        # Calculate PE_w1' and PE_w2'
        pe1_prime: float = cls._compute_potential_energy(seq_to_compare, seq1_prime)
        pe2_prime: float = cls._compute_potential_energy(seq_to_compare, seq2_prime)

        # Let temp = PE_w + KE_w - PE_w1' - PE_w2'
        temp: float = molecule.potential_energy - pe1_prime - pe2_prime

        # Create a Boolean variable success
        success: bool = False

        if temp >= 0:
            success: bool = True

            # Get k randomly in interval [0, 1]
            loss_rate: float = np.random.uniform(0, 1)

            # KE_w1' = temp * k
            ke1_prime: float = temp * loss_rate

            # KE_w2' = temp * (1 - k)
            ke2_prime: float = temp * (1 - loss_rate)

            # Assign w_1', PE_w1' and KE_w1' to the profile of M_1',
            # and w_2', PE_w2' and KE_w2' to the profile of M_2'
            seq1.genes = seq1_prime.genes
            seq1.fitness = pe1_prime
            molecule1_prime.potential_energy = pe1_prime
            molecule1_prime.kinetic_energy = ke1_prime

            seq2.genes = seq2_prime.genes
            seq2.fitness = pe2_prime
            molecule2_prime.potential_energy = pe2_prime
            molecule2_prime.kinetic_energy = ke2_prime
        elif temp + buffer >= 0:
            success: bool = True

            # Get m_1, m_2, m_3 and m_4 independently randomly in interval [0, 1].
            m1, m2 = np.random.uniform(0, 1), np.random.uniform(0, 1)
            m3, m4 = np.random.uniform(0, 1), np.random.uniform(0, 1)

            # KE_w1' = (temp + buffer) * m1 * m2
            ke1_prime: float = (temp + buffer) * m1 * m2
            # KE_w2' = (temp + buffer - KE_w1') * m3 * m4
            ke2_prime: float = (temp + buffer - ke1_prime) * m3 * m4

            # Update buffer = temp + buffer - KE_w1' - KE_w2'
            buffer: float = temp + buffer - ke1_prime - ke2_prime

            # Assign w_1', PE_w1' and KE_w1' to the profile of M_1',
            # and w_2', PE_w2' and KE_w2' to the profile of M_2'
            seq1.genes = seq1_prime.genes
            seq1.fitness = pe1_prime
            molecule1_prime.potential_energy = pe1_prime
            molecule1_prime.kinetic_energy = ke1_prime

            seq2.genes = seq2_prime.genes
            seq2.fitness = pe2_prime
            molecule2_prime.potential_energy = pe2_prime
            molecule2_prime.kinetic_energy = ke2_prime

        # Output M_1' and M_2', success and buffer
        return seq1, seq2, success, buffer

    @classmethod  # Input: Molecule, M_1, M_2 with their profiles.
    def ineffective_intermolecular_collision(cls, seq_to_compare: Sequence,
                                             seq1: Sequence, seq2: Sequence) -> tuple[Sequence, Sequence]:
        """
            ~ Colisión Intermolecular Ineficaz ~
        Una colisión intermolecular ineficaz describe la situación en la que dos
        moléculas chocan entre sí y luego rebotan.

            * El efecto del cambio de energía de las moléculas es similar al de una colisión
            ineficaz en la pared, pero esta reacción elemental involucra más de una molécula
            y no se obtiene Energía cinética del búfer central.
        """
        molecule1: Molecule = seq1.to_molecule_instance()
        molecule2: Molecule = seq2.to_molecule_instance()

        # Obtain w_1' = Neighbor(w_1) and w_2' = Neighbor(w_2)
        seq1_prime: Sequence = seq1.__copy__()
        seq2_prime: Sequence = seq2.__copy__()

        molecule1_prime: Molecule = seq1_prime.to_molecule_instance()
        molecule2_prime: Molecule = seq2_prime.to_molecule_instance()

        SimpleMutator.mutate_seqs_genes([seq1_prime, seq2_prime])

        # Calculate PE_w1' and PE_w2'
        pe1_prime: float = cls._compute_potential_energy(seq_to_compare, seq1_prime)
        pe2_prime: float = cls._compute_potential_energy(seq_to_compare, seq2_prime)

        # Let temp = (PE_w1 + KE_w1 + PE_w2 + KE_w2) - (PE_w1' - PE_w2')
        temp: float = (molecule1.total_energy + molecule2.total_energy) - \
                      (pe1_prime + pe2_prime)

        if temp >= 0:
            # Get p randomly in interval [0, 1]
            loss_rate: float = np.random.uniform(0, 1)

            # KE_w1' = temp * p
            ke1_prime: float = temp * loss_rate
            # KE_w2' = temp * (1 - p)
            ke2_prime: float = temp * (1 - loss_rate)

            # Update the profile of M_1 by w_1 = w_1', PE_w1 = PE_w1'
            # and KE_w1 = KE_w1', and the profile of M_2 by w_2 = w_2',
            # PE_w2 = PE_w2' and KE_w2 = KE_w2'.
            seq1.genes = seq1_prime.genes
            seq1.fitness = pe1_prime
            molecule1_prime.potential_energy = pe1_prime
            molecule1_prime.kinetic_energy = ke1_prime

            seq2.genes = seq2_prime.genes
            seq2.fitness = pe2_prime
            molecule2_prime.potential_energy = pe2_prime
            molecule2_prime.kinetic_energy = ke2_prime

        # Output M_1 and M_2
        return seq1, seq2

    @classmethod  # Input: molecules M1, M2, with their profiles.
    def synthesis(cls, seq_to_compare: Sequence, seq1: Sequence, seq2: Sequence) -> tuple[Sequence, bool]:
        """
            ~ Síntesis ~
        Una síntesis representa más de una molécula (suponga dos moléculas)
        que chocan y se combinan.
        """
        molecule1: Molecule = seq1.to_molecule_instance()
        molecule2: Molecule = seq2.to_molecule_instance()

        # Obtain w' from w_1 and w_2
        seq1_prime: Sequence = seq1.__copy__()
        seq2_prime: Sequence = seq2.__copy__()

        seq_prime, _ = BCMutator.generate_mutated_seqs([seq1_prime, seq2_prime])

        # Calculate PE_w'
        pe_prime: float = cls._compute_potential_energy(seq_to_compare, seq_prime)

        # Create a Boolean variable success
        success: bool = False

        # Create a new molecule M'
        molecule_prime: Molecule = seq_prime.to_molecule_instance()

        # if PE_w1 + PE_w2 + KE_w1 + KE_w2 >= PE_w'
        molecules_total_energy: float = molecule1.total_energy + molecule2.total_energy

        if molecules_total_energy >= pe_prime:
            success: bool = True

            # KE_w' = PE_w1 + PE_w2 + KE_w1 + KE_w2 - PE_w'
            ke_prime: float = molecules_total_energy - pe_prime

            # Assign w'. PE_w' and KE_w' to the profile of M'
            seq_prime.fitness = pe_prime
            molecule_prime.potential_energy = pe_prime
            molecule_prime.kinetic_energy = ke_prime

        # Output M' and success
        # return new_molecule, success
        return seq_prime, success

    @classmethod
    def _compute_potential_energy(cls, seq_to_compare: Sequence, seq: Sequence) -> float:
        FitnessCalculator.compute_seqs_fitness(seq_to_compare, [seq])
        seq_to_compare.fitness = seq.fitness
        return seq.fitness
