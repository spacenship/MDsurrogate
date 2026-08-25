"""Explicit unit metadata.

Every physical tensor that crosses a module boundary carries its units through a
:class:`UnitMetadata` attached to the batch, rather than relying on a convention
remembered in a docstring. mdCATH's own unit attributes are read from the file
and checked against :data:`MDCATH_UNITS` by the adapter -- they are never
assumed.
"""

from __future__ import annotations

from dataclasses import dataclass

# Physical constants, in the unit system below.
# Boltzmann constant in kcal/mol/K (CHARMM/AMBER convention).
BOLTZMANN_KCAL_PER_MOL_PER_K = 0.0019872041

ANGSTROM_PER_NM = 10.0


@dataclass(frozen=True)
class UnitMetadata:
    """Units of the physical quantities in a batch.

    Attributes:
        length: unit of ``positions`` and of any displacement vector.
        force: unit of ``forces`` and of every predicted force/torque-per-length.
        energy: unit of the energy head's output.
        temperature: unit of ``temperature``.

    The default is mdCATH's native system, so no conversion happens by default
    and a conversion bug cannot hide behind an identity transform.
    """

    length: str = "angstrom"
    force: str = "kcal/mol/angstrom"
    energy: str = "kcal/mol"
    temperature: str = "kelvin"

    def __post_init__(self) -> None:
        if self.length not in ("angstrom", "nanometer"):
            raise ValueError(f"unsupported length unit: {self.length!r}")
        if self.temperature != "kelvin":
            raise ValueError(f"unsupported temperature unit: {self.temperature!r}")

    @property
    def torque(self) -> str:
        """Unit of a residue torque, i.e. length x force."""
        if self.length == "angstrom":
            return "kcal/mol"
        return "kcal/mol*nanometer/angstrom"

    def kT(self, temperature_kelvin: float) -> float:
        """Thermal energy at ``temperature_kelvin``, in :attr:`energy` units."""
        if self.energy != "kcal/mol":
            raise ValueError(f"kT not defined for energy unit {self.energy!r}")
        return BOLTZMANN_KCAL_PER_MOL_PER_K * float(temperature_kelvin)


#: Units as recorded in the mdCATH HDF5 files (verified by reading the
#: ``unit`` attributes of ``coords`` and ``forces``; see docs/).
MDCATH_UNITS = UnitMetadata(
    length="angstrom",
    force="kcal/mol/angstrom",
    energy="kcal/mol",
    temperature="kelvin",
)

#: The five simulation temperatures present in mdCATH, in kelvin. These are
#: literal group names in the files, not an assumption.
MDCATH_TEMPERATURES_K = (320, 348, 379, 413, 450)

#: Physical time between consecutive saved frames: **1 ns**.
#:
#: This is NOT readable from the shards -- they carry no per-frame timestamp --
#: it comes from the mdCATH publication. It is kept as a named constant rather
#: than sprinkled through the code so that the one place it could be wrong is
#: obvious. Phase 1 does not use it (force supervision is per frame); Phase 2's
#: 1/2/4/8/16 ns lags are 1/2/4/8/16 frames apart.
MDCATH_PS_PER_FRAME = 1000.0
