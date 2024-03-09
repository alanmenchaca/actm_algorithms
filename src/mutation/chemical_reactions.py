from dataclasses import dataclass, field

import numpy as np

from mutation.backpack_mutator import BackpackCrosserMutator
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
    _sm: SimpleMutator = field(default=None)
    _bcm: BackpackCrosserMutator = field(default=None)
    _fc: FitnessCalculator = field(default=None)

    _main_seq: Sequence = field(default=None)
    _seq1: Sequence = field(default=None)
    _seq2: Sequence = field(default=None)
    _seqs: list[Sequence] = field(default=None)

    _buffer: float = field(default=0.01)

    def __post_init__(self):
        self._sm = SimpleMutator()
        self._bcm = BackpackCrosserMutator()
        self._fc = FitnessCalculator()

    def collide_molecules(self, seq_to_compare: Sequence, seqs: list[Sequence],
                          num_collisions: int = 100) -> None:
        for i in range(num_collisions):
            self._main_seq = seq_to_compare
            self._seqs = seqs

            # choose two random sequences
            self._seq1, self._seq2 = np.random.choice(seqs, 2, replace=False)

            # select a random collision type
            collision_type: int = np.random.randint(0, 4)
            self._do_collision(collision_type)

    def _do_collision(self, collision_type: int) -> None:
        if collision_type == 0:
            _, self._buffer = self.ineffective_collision_against_the_wall(self._main_seq,
                                                                          self._seq1, self._buffer)
        elif collision_type == 1:
            generated_seq1, generated_seq2, success, self._buffer = self \
                .decomposition(self._main_seq, self._seq1, self._buffer)
            if success:
                self._seqs.remove(self._seq1)
                self._seqs.append(generated_seq1)
                self._seqs.append(generated_seq2)
        elif collision_type == 2:
            _, _ = self.ineffective_intermolecular_collision(self._main_seq, self._seq1, self._seq2)
        else:
            generated_seq, success = self.synthesis(self._main_seq, self._seq1, self._seq2)
            if success:
                self._seqs.remove(self._seq1)
                self._seqs.remove(self._seq2)
                self._seqs.append(generated_seq)

    # Input: A molecule M with its profile and central energy buffer.
    def ineffective_collision_against_the_wall(self, seq_to_compare: Sequence, seq: Sequence,
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
        new_seq_mutated: Sequence = seq.__copy__()
        self._sm.mutate_seqs_genes([new_seq_mutated])

        # Calculate PE_w'
        new_potential_energy: float = self._compute_potential_energy(seq_to_compare, new_seq_mutated)

        # Change is allowed if: PE_w + KE_w >= PE_w'
        if molecule.total_energy >= new_potential_energy:
            # Get q randomly in interval [KELossRate, 1]
            ke_loss_rate: float = np.random.uniform(0, 1)

            # KE_w' = (PE_w + KE_w - PE_w') * q
            new_kinetic_energy: float = (molecule.total_energy - new_potential_energy) * ke_loss_rate

            # buffer = buffer + (PE_w + KE_w - PE_w') * (1 - q)
            buffer: float = (buffer + molecule.total_energy - new_potential_energy) * (1 - ke_loss_rate)

            # Update the profile o M by w = w'. PE_w = PE_w' and KE_w = KE_w'
            seq.genes = new_seq_mutated.genes
            seq.fitness = new_potential_energy
            molecule.potential_energy = new_potential_energy
            molecule.kinetic_energy = new_kinetic_energy

        # Output M and buffer
        return seq, buffer

    # Input: A Molecule M with its profile and the central energy buffer.
    def decomposition(self, seq_to_compare: Sequence, seq: Sequence,
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
        new_first_seq: Sequence = seq.__copy__()
        new_second_seq: Sequence = seq.__copy__()

        new_first_molecule: Molecule = new_first_seq.to_molecule_instance()
        new_second_molecule: Molecule = new_second_seq.to_molecule_instance()

        # Obtain w_1' and w_2' from w
        new_first_seq_mutated: Sequence = new_first_seq.__copy__()
        new_second_seq_mutated: Sequence = new_second_seq.__copy__()

        self._sm.mutate_seqs_genes([new_first_seq_mutated, new_second_seq_mutated])

        # Calculate PE_w1' and PE_w2'
        new_first_potential_energy: float = self._compute_potential_energy(seq_to_compare, new_first_seq_mutated)
        new_second_potential_energy: float = self._compute_potential_energy(seq_to_compare, new_second_seq_mutated)

        # Let temp = PE_w + KE_w - PE_w1' - PE_w2'
        temp: float = molecule.potential_energy - new_first_potential_energy - new_second_potential_energy

        # Create a Boolean variable success
        success: bool = False

        if temp >= 0:
            success: bool = True

            # Get k randomly in interval [0, 1]
            loss_rate: float = np.random.uniform(0, 1)

            # KE_w1' = temp * k
            new_first_kinetic_energy: float = temp * loss_rate

            # KE_w2' = temp * (1 - k)
            new_second_kinetic_energy: float = temp * (1 - loss_rate)

            # Assign w_1', PE_w1' and KE_w1' to the profile of M_1',
            # and w_2', PE_w2' and KE_w2' to the profile of M_2'
            new_first_seq.genes = new_first_seq_mutated.genes
            new_first_seq.fitness = new_first_potential_energy
            new_first_molecule.potential_energy = new_first_potential_energy
            new_first_molecule.kinetic_energy = new_first_kinetic_energy

            new_second_seq.genes = new_second_seq_mutated.genes
            new_second_seq.fitness = new_second_potential_energy
            new_second_molecule.potential_energy = new_second_potential_energy
            new_second_molecule.kinetic_energy = new_second_kinetic_energy
        elif temp + buffer >= 0:
            success: bool = True

            # Get m_1, m_2, m_3 and m_4 independently randomly in interval [0, 1].
            m1, m2 = np.random.uniform(0, 1), np.random.uniform(0, 1)
            m3, m4 = np.random.uniform(0, 1), np.random.uniform(0, 1)

            # KE_w1' = (temp + buffer) * m1 * m2
            new_first_kinetic_energy: float = (temp + buffer) * m1 * m2
            # KE_w2' = (temp + buffer - KE_w1') * m3 * m4
            new_second_kinetic_energy: float = (temp + buffer - new_first_kinetic_energy) * m3 * m4

            # Update buffer = temp + buffer - KE_w1' - KE_w2'
            buffer: float = temp + buffer - new_first_kinetic_energy - new_second_kinetic_energy

            # Assign w_1', PE_w1' and KE_w1' to the profile of M_1',
            # and w_2', PE_w2' and KE_w2' to the profile of M_2'
            new_first_seq.genes = new_first_seq_mutated.genes
            new_first_seq.fitness = new_first_potential_energy
            new_first_molecule.potential_energy = new_first_potential_energy
            new_first_molecule.kinetic_energy = new_first_kinetic_energy

            new_second_seq.genes = new_second_seq_mutated.genes
            new_second_seq.fitness = new_second_potential_energy
            new_second_molecule.potential_energy = new_second_potential_energy
            new_second_molecule.kinetic_energy = new_second_kinetic_energy

        # Output M_1' and M_2', success and buffer
        return new_first_seq, new_second_seq, success, buffer

    # Input: Molecule, M_1, M_2 with their profiles.
    def ineffective_intermolecular_collision(self, seq_to_compare: Sequence,
                                             seq1: Sequence, seq2: Sequence) -> tuple[Sequence, Sequence]:
        """
            ~ Colisión Intermolecular Ineficaz ~
        Una colisión intermolecular ineficaz describe la situación en la que dos
         moléculas chocan entre sí y luego rebotan.

            * El efecto del cambio de energía de las moléculas es similar al de una colisión
            ineficaz en la pared, pero esta reacción elemental involucra más de una molécula
            y no se obtiene Energía cinética del búfer central.
        """
        first_molecule: Molecule = seq1.to_molecule_instance()
        second_molecule: Molecule = seq2.to_molecule_instance()

        # Obtain w_1' = Neighbor(w_1) and w_2' = Neighbor(w_2)
        new_first_seq: Sequence = seq1.__copy__()
        new_second_seq: Sequence = seq2.__copy__()

        new_first_molecule: Molecule = new_first_seq.to_molecule_instance()
        new_second_molecule: Molecule = new_second_seq.to_molecule_instance()

        self._sm.mutate_seqs_genes([new_first_seq, new_second_seq])

        # Calculate PE_w1' and PE_w2'
        new_first_potential_energy: float = self._compute_potential_energy(seq_to_compare, new_first_seq)
        new_second_potential_energy: float = self._compute_potential_energy(seq_to_compare, new_second_seq)

        # Let temp = (PE_w1 + KE_w1 + PE_w2 + KE_w2) - (PE_w1' - PE_w2')
        temp: float = (first_molecule.total_energy + second_molecule.total_energy) - \
                      (new_first_potential_energy + new_second_potential_energy)

        if temp >= 0:
            # Get p randomly in interval [0, 1]
            loss_rate: float = np.random.uniform(0, 1)

            # KE_w1' = temp * p
            new_first_kinetic_energy: float = temp * loss_rate
            # KE_w2' = temp * (1 - p)
            new_second_kinetic_energy: float = temp * (1 - loss_rate)

            # Update the profile of M_1 by w_1 = w_1', PE_w1 = PE_w1'
            # and KE_w1 = KE_w1', and the profile of M_2 by w_2 = w_2',
            # PE_w2 = PE_w2' and KE_w2 = KE_w2'.
            seq1.genes = new_first_seq.genes
            seq1.fitness = new_first_potential_energy
            new_first_molecule.potential_energy = new_first_potential_energy
            new_first_molecule.kinetic_energy = new_first_kinetic_energy

            seq2.genes = new_second_seq.genes
            seq2.fitness = new_second_potential_energy
            new_second_molecule.potential_energy = new_second_potential_energy
            new_second_molecule.kinetic_energy = new_second_kinetic_energy

        # Output M_1 and M_2
        return seq1, seq2

    # Input: molecules M1, M2, with their profiles.
    def synthesis(self, seq_to_compare: Sequence, seq1: Sequence, seq2: Sequence) -> tuple[Sequence, bool]:
        """
            ~ Síntesis ~
        Una síntesis representa más de una molécula (suponga dos moléculas)
         que chocan y se combinan.
        """
        first_molecule: Molecule = seq1.to_molecule_instance()
        second_molecule: Molecule = seq2.to_molecule_instance()

        # Obtain w' from w_1 and w_2
        new_first_seq: Sequence = seq1.__copy__()
        new_second_seq: Sequence = seq2.__copy__()

        new_mutated_seq, _ = self._bcm \
            .generate_mutated_population([new_first_seq, new_second_seq])

        # Calculate PE_w'
        new_potential_energy: float = self._compute_potential_energy(seq_to_compare, new_mutated_seq)

        # Create a Boolean variable success
        success: bool = False

        # Create a new molecule M'
        new_molecule: Molecule = new_mutated_seq.to_molecule_instance()

        # if PE_w1 + PE_w2 + KE_w1 + KE_w2 >= PE_w'
        molecules_total_energy: float = first_molecule.total_energy + second_molecule.total_energy

        if molecules_total_energy >= new_potential_energy:
            success: bool = True

            # KE_w' = PE_w1 + PE_w2 + KE_w1 + KE_w2 - PE_w'
            new_kinetic_energy: float = molecules_total_energy - new_potential_energy

            # Assign w'. PE_w' and KE_w' to the profile of M'
            new_mutated_seq.fitness = new_potential_energy
            new_molecule.potential_energy = new_potential_energy
            new_molecule.kinetic_energy = new_kinetic_energy

        # Output M' and success
        # return new_molecule, success
        return new_mutated_seq, success

    def _compute_potential_energy(self, seq_to_compare: Sequence, seq: Sequence) -> float:
        self._fc.set_main_seq(seq_to_compare)
        self._fc.set_secondary_seq(seq)

        potential_energy: float = self._fc \
            .compute_fitness_between_main_and_secondary_seqs()
        seq.fitness = potential_energy

        return potential_energy
