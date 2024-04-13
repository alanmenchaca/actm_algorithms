from dataclasses import field, dataclass


@dataclass
class Molecule:
    potential_energy: int = field(default=0, init=False)
    kinetic_energy: float = field(default=0.0, init=False)

    @property
    def total_energy(self) -> float:
        return self.potential_energy + self.kinetic_energy
