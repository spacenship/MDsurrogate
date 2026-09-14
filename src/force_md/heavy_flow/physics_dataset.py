"""Leakage-safe frame dataset for direct Stage 3 atom-force physics."""

from __future__ import annotations

import dataclasses
from collections import OrderedDict
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from ..data.adapters.mdcath import MdCathDataset
from ..heavy.domain_topology import load_domain_topology
from .data import bonds_from_topology, collate_heavy_flow, from_mdcath_example
from .types import HeavyFlowSample
from .history import HistoryConfig
from ..data.adapters.lag_pairs import exact_lag_frames
from ..data.units import MDCATH_PS_PER_FRAME

__all__ = ["PhysicsFrameKey", "PhysicsSplitManifest", "split_physics_frames", "HeavyFlowPhysicsFrameDataset", "HeavyFlowTemporalPairDataset", "build_physics_splits", "collate_heavy_flow"]


@dataclass(frozen=True, order=True)
class PhysicsFrameKey:
    domain: str
    temperature: str
    replica: str
    frame: int

    @classmethod
    def from_value(cls, value: Any) -> "PhysicsFrameKey":
        if isinstance(value, cls):
            return value
        if isinstance(value, dict):
            return cls(str(value["domain"]), str(value.get("temperature", value.get("temp"))), str(value["replica"]), int(value["frame"]))
        if len(value) != 4:
            raise ValueError("frame key must be (domain, temperature, replica, frame)")
        return cls(str(value[0]), str(value[1]), str(value[2]), int(value[3]))

    def as_tuple(self) -> tuple[str, str, str, int]:
        return self.domain, self.temperature, self.replica, self.frame

    def as_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclass
class PhysicsSplitManifest:
    train: list[PhysicsFrameKey]
    same_domain_validation: list[PhysicsFrameKey]
    unseen_domain_validation: list[PhysicsFrameKey]
    seed: int = 0
    temporal_block_size: int = 16
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.train = [PhysicsFrameKey.from_value(x) for x in self.train]
        self.same_domain_validation = [PhysicsFrameKey.from_value(x) for x in self.same_domain_validation]
        self.unseen_domain_validation = [PhysicsFrameKey.from_value(x) for x in self.unseen_domain_validation]
        self.assert_disjoint()

    def assert_disjoint(self) -> None:
        sets = [set(self.train), set(self.same_domain_validation), set(self.unseen_domain_validation)]
        if sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2]:
            raise ValueError("physics split manifest contains overlapping frame keys")

    def to_dict(self) -> dict[str, Any]:
        return {"train": [x.as_dict() for x in self.train], "same_domain_validation": [x.as_dict() for x in self.same_domain_validation], "unseen_domain_validation": [x.as_dict() for x in self.unseen_domain_validation], "seed": self.seed, "temporal_block_size": self.temporal_block_size, "metadata": self.metadata}

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n")

    @classmethod
    def load(cls, path: str | Path) -> "PhysicsSplitManifest":
        return cls(**json.loads(Path(path).read_text()))


def split_physics_frames(indices: Iterable[Any], *, same_domain_fraction: float = 0.2, unseen_domain_fraction: float = 0.2, temporal_block_size: int = 16, seed: int = 0) -> PhysicsSplitManifest:
    """Split complete frame indices by domain and non-overlapping time blocks."""
    if not 0 <= same_domain_fraction < 1 or not 0 <= unseen_domain_fraction < 1:
        raise ValueError("validation fractions must be in [0,1)")
    if temporal_block_size < 1:
        raise ValueError("temporal_block_size must be positive")
    keys = sorted({PhysicsFrameKey.from_value(value) for value in indices})
    if not keys:
        raise ValueError("cannot split an empty frame index")
    rng = np.random.default_rng(seed)
    domains = sorted({key.domain for key in keys})
    unseen_count = 0 if unseen_domain_fraction == 0 else max(1, int(round(len(domains) * unseen_domain_fraction)))
    unseen_count = min(unseen_count, max(0, len(domains) - 1))
    unseen_domains = set(rng.permutation(domains)[:unseen_count].tolist())
    unseen = [key for key in keys if key.domain in unseen_domains]
    remaining = [key for key in keys if key.domain not in unseen_domains]
    grouped: dict[tuple[str, str, str, int], list[PhysicsFrameKey]] = {}
    for key in remaining:
        grouped.setdefault((key.domain, key.temperature, key.replica, key.frame // temporal_block_size), []).append(key)
    block_keys = sorted(grouped)
    validation_blocks: set[tuple[str, str, str, int]] = set()
    if same_domain_fraction > 0 and block_keys:
        validation_count = min(max(1, int(round(len(block_keys) * same_domain_fraction))), max(0, len(block_keys) - 1))
        if validation_count:
            validation_blocks = {block_keys[int(i)] for i in rng.permutation(len(block_keys))[:validation_count]}
    same_domain = [key for key in remaining if (key.domain, key.temperature, key.replica, key.frame // temporal_block_size) in validation_blocks]
    same_set = set(same_domain)
    train = [key for key in remaining if key not in same_set]
    return PhysicsSplitManifest(
        train=train, same_domain_validation=same_domain, unseen_domain_validation=unseen,
        seed=seed, temporal_block_size=temporal_block_size,
        metadata={"split_type": "physics_frame", "unseen_domains": sorted(unseen_domains), "same_domain_validation_blocks": [list(block) for block in sorted(validation_blocks)], "train_domains": sorted({key.domain for key in train}), "same_domain_validation_domains": sorted({key.domain for key in same_domain}), "counts": {"train": len(train), "same_domain_validation": len(same_domain), "unseen_domain_validation": len(unseen)}},
    )


class HeavyFlowPhysicsFrameDataset(Dataset[HeavyFlowSample]):
    """Direct current-force frame adapter over the existing mdCATH reader."""

    def __init__(self, reader: MdCathDataset, keys: Sequence[Any], *, history_config: Optional[HistoryConfig] = None, history_provider: Optional[Callable[[PhysicsFrameKey, np.ndarray], torch.Tensor]] = None):
        self.reader = reader
        self.history_config = history_config or HistoryConfig()
        if history_provider is not None:
            raise ValueError("unverified history_provider is unsupported; use raw-frame HistoryConfig")
        self.keys = []
        self.excluded_keys = []
        for value in keys:
            key = PhysicsFrameKey.from_value(value)
            try:
                frames = self.history_config.frames(key.frame)
                self._check_frames(key, frames)
            except ValueError:
                self.excluded_keys.append(key)
                continue
            self.keys.append(key)
        self._bond_cache: OrderedDict[str, tuple[torch.Tensor, torch.Tensor]] = OrderedDict()

    def _check_frames(self, key, frames):
        bad = getattr(self.reader, "coord_quarantine", {}).get(key.domain, {}).get(f"{key.temperature}/{key.replica}", set())
        if any(frame in bad for frame in frames):
            raise ValueError("history/current/future touches a coordinate-quarantined frame")

    def _example(self, key, frame):
        self._check_frames(key, (frame,))
        arrays = self.reader.load_frame_arrays(key.domain, key.temperature, key.replica, frame)
        return self.reader.build_example(key.domain, key.temperature, key.replica, frame, *arrays)

    @staticmethod
    def _aligned_positions(reference, other):
        for name in ("atomic_number", "atom_name_id", "atom_to_residue", "is_backbone", "is_cap"):
            if not torch.equal(getattr(reference.batch.atoms, name), getattr(other.batch.atoms, name)):
                raise ValueError(f"history/future atom mapping differs: {name}")
        if not torch.equal(reference.batch.residues.residue_type, other.batch.residues.residue_type):
            raise ValueError("history/future residue mapping differs")
        positions = other.batch.atoms.positions
        if positions.shape != reference.batch.atoms.positions.shape:
            raise ValueError("history/future atom count differs")
        return positions

    def __len__(self) -> int:
        return len(self.keys)

    @property
    def domains(self) -> list[str]:
        return [key.domain for key in self.keys]

    def __getitem__(self, index: int) -> HeavyFlowSample:
        key = self.keys[index]
        example = self._example(key, key.frame)
        # load_frame_arrays is in raw/topology order; build_example is the
        # authoritative represented heavy-atom order used by the model.
        current = example.batch.atoms.positions
        frames = self.history_config.frames(key.frame)
        positions = {key.frame: current}
        for frame in set(frames) - {key.frame}:
            positions[frame] = self._aligned_positions(example, self._example(key, frame))
        history = torch.stack([positions[frame] for frame in frames])
        if key.domain not in self._bond_cache:
            topology = load_domain_topology(self.reader.config.data_dir, key.domain, represented_scope=self.reader.config.represented_scope)
            self._bond_cache[key.domain] = bonds_from_topology(topology, topology.raw_to_batch, device=current.device)
            if len(self._bond_cache) > 8:
                self._bond_cache.popitem(last=False)
        self._bond_cache.move_to_end(key.domain)
        bonds, bond_type = self._bond_cache[key.domain]
        future_frame = key.frame + getattr(self, "future_lag_frames", 0)
        future = None if future_frame == key.frame else self._aligned_positions(example, self._example(key, future_frame))
        sample = from_mdcath_example(example, x_history=history, x_future=future, bond_index=bonds, bond_type=bond_type, allow_identity_future=future is None)
        provenance = dataclasses.replace(sample.condition.provenance, history_frames=(frames,),
                                         future_frame=(future_frame,) if future is not None else (),
                                         preprocessing_version="heavy_flow_v2", time_unit="frame",
                                         time_per_frame=getattr(self.reader.config, "ps_per_frame", None) or MDCATH_PS_PER_FRAME)
        sample.condition.provenance = provenance
        sample.targets.provenance = provenance
        if future is not None:
            sample.condition.lag.fill_(self.future_lag_frames)
        sample.validate()
        return sample


class HeavyFlowTemporalPairDataset(HeavyFlowPhysicsFrameDataset):
    """Same-trajectory raw-frame pairs with the same history used by Stage 3.

    Uses the existing exact-lag conversion. Decoder lag is in stored frames;
    provenance records picoseconds per frame (mdCATH: 1000 ps).
    """
    def __init__(self, reader, keys, *, history_config=None, future_lag_frames=1):
        spacing = float(reader.config.ps_per_frame if reader.config.ps_per_frame is not None else MDCATH_PS_PER_FRAME)
        self.future_lag_frames = exact_lag_frames(float(future_lag_frames) * spacing, spacing)
        super().__init__(reader, keys, history_config=history_config)
        counts = {}
        valid = []
        for key in self.keys:
            trajectory = (key.domain, key.temperature, key.replica)
            if trajectory not in counts:
                group = reader._open(key.domain)[key.domain][key.temperature][key.replica]
                counts[trajectory] = int(group.attrs["numFrames"])
            future = key.frame + self.future_lag_frames
            try:
                self._check_frames(key, (future,))
            except ValueError:
                self.excluded_keys.append(key)
                continue
            if future < counts[trajectory]:
                valid.append(key)
            else:
                self.excluded_keys.append(key)
        self.keys = valid


def build_physics_splits(reader: MdCathDataset, **kwargs: Any) -> PhysicsSplitManifest:
    return split_physics_frames(reader.index, **kwargs)
