"""Raw-trajectory lag pairs -- the Phase 1.5 transition dataset.

One item is a **transition**::

    (q_{t-1}, q_t)  ->  q_{t+lag}          history_length = 2 (default)

Four rules decide whether this dataset is measuring what it claims to, and each
of them is a place where a plausible-looking implementation would be wrong:

**Lags are computed from the raw frame index, never from Phase 1's index.**
Phase 1 samples ``frames_per_trajectory`` frames evenly over a ~450-frame
trajectory, which puts *12-13 frames* between consecutive index entries. Two
adjacent Phase 1 entries are therefore ~12-13 ns apart, not 1 ns. Reusing that
index for a 1 ns lag would be off by an order of magnitude and would look
entirely reasonable in the code. Here ``lag_frames = lag_ps / ps_per_frame`` and
the frame numbers are the file's own.

**The division must be exact.** ``lag_ps / ps_per_frame`` is required to be an
integer; a lag that does not land on a stored frame raises rather than rounding.
Rounding 1500 ps to "1 or 2 frames" would silently relabel the physical time of
every pair in the manifest.

**A pair never crosses a trajectory.** All of history, current and future come
from one ``(domain, temperature, replica)`` trajectory. Crossing a replica
boundary would pair two independent simulations and call the difference dynamics.

**The split stays Phase 1's, by domain.** Consecutive frames of one protein are
strongly correlated, so re-splitting by frame would measure memorisation. The
domain lists are restored from the Phase 1 run and checked for overlap.

The index is built from HDF5 *attributes* only (``numFrames``), so building a
manifest over 907 domains touches none of the 623 GB of coordinates; frames are
read lazily in ``__getitem__``.

Both quarantines are reused unchanged through :class:`MdCathDataset`: a
coordinate-quarantined frame disqualifies every pair that would touch it (as
history, current or future), and force-quarantined trajectories are excluded by
default so that the oracle arm sees exactly the pairs the production arms do.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field, replace
from typing import Iterable, Optional, Sequence

import numpy as np
import torch
from torch import Tensor

from ..collate import collate_batches, collate_frame_geometries
from ..contracts import FrameGeometry, HierarchicalProteinBatch
from .mdcath import MdCathConfig, MdCathDataset, TrainingExample, _Topology, split_domains

__all__ = [
    "exact_lag_frames",
    "LagPairConfig",
    "LagPair",
    "LagPairManifest",
    "LagPairExample",
    "LagPairBatch",
    "LagPairDataset",
    "build_lag_pair_manifest",
    "collate_lag_pairs",
    "restore_phase1_split",
    "load_phase1_split",
]

#: Manifest format version. Bumped when the row layout changes, so an old file
#: fails loudly instead of being reinterpreted under new column meanings.
MANIFEST_VERSION = 1


def exact_lag_frames(lag_ps: float, ps_per_frame: float) -> int:
    """Frames in ``lag_ps``, or raise.

    Args:
        lag_ps: physical lag in picoseconds.
        ps_per_frame: physical time between stored frames. For mdCATH this is
            1000.0 ps (:data:`force_md.data.units.MDCATH_PS_PER_FRAME`) -- from
            the publication, not from the files, which carry no timestamp.

    Raises:
        ValueError: if the lag is not an exact multiple of the frame spacing.
            Silently rounding would attach a physical time to a pair that does
            not have it, and every downstream number would inherit the error.
    """
    if ps_per_frame is None or ps_per_frame <= 0:
        raise ValueError(f"ps_per_frame must be positive, got {ps_per_frame!r}")
    if lag_ps <= 0:
        raise ValueError(f"lag_ps must be positive, got {lag_ps!r}")
    quotient = lag_ps / ps_per_frame
    rounded = round(quotient)
    if abs(quotient - rounded) > 1e-9:
        raise ValueError(
            f"lag {lag_ps} ps is {quotient:.6f} frames at {ps_per_frame} ps/frame, "
            "which is not an integer. mdCATH stores frames, not a continuous "
            "trajectory, so a lag that does not land on a stored frame cannot be "
            "built. Choose a multiple of the frame spacing rather than rounding."
        )
    return int(rounded)


@dataclass(frozen=True)
class LagPairConfig:
    """Configuration of the lag-pair index.

    Args:
        mdcath: the Phase 1 adapter config. Its ``data_dir``, quarantine paths,
            ``represented_scope``, ``temperatures``/``replicas``, ``max_residues``
            and ``ps_per_frame`` are all honoured. Its ``frames_per_trajectory``
            is **ignored** and must be, for the reason in the module docstring.
        lags_ps: physical lags. Default ``(1000, 4000)`` ps = 1 ns and 4 ns.
        history_length: frames of state given to the model, counting the current
            one. 2 means ``(t-1, t)``; 1 means ``(t,)`` and is a diagnostic.
        require_current_force_labels: keep only pairs whose current frame has
            usable force labels. True by default so that arms A-E -- including
            the ground-truth-force oracle -- run on exactly the same pairs; a
            production-only manifest (no oracle arm) may set it False.
        pairs_per_trajectory: cap per trajectory and lag, selected by
            ``selection``. None keeps every candidate.
        max_trajectories_per_domain / max_domains / max_pairs: smoke caps.
        selection: ``"even"`` spreads the picks over the trajectory with a fixed
            stride; ``"random"`` draws them with ``seed``. Even is the default
            because it makes two manifests at different sizes nested and
            comparable.
        seed: recorded in the manifest always; used only by ``selection="random"``.
    """

    mdcath: MdCathConfig
    lags_ps: tuple[float, ...] = (1000.0, 4000.0)
    history_length: int = 2
    require_current_force_labels: bool = True
    pairs_per_trajectory: Optional[int] = None
    max_trajectories_per_domain: Optional[int] = None
    max_domains: Optional[int] = None
    max_pairs: Optional[int] = None
    selection: str = "even"
    seed: int = 0

    def __post_init__(self) -> None:
        if self.history_length not in (1, 2):
            raise ValueError(
                f"history_length must be 1 or 2, got {self.history_length}. Longer "
                "histories are a Phase 2 question; this probe fixes the ablation at "
                "2 and allows 1 only as a diagnostic."
            )
        if self.selection not in ("even", "random"):
            raise ValueError(f"selection must be 'even' or 'random', got {self.selection!r}")
        if not self.lags_ps:
            raise ValueError("lags_ps is empty")
        if self.mdcath.ps_per_frame is None:
            raise ValueError(
                "MdCathConfig.ps_per_frame is None. Phase 1 did not need it -- force "
                "supervision is per frame -- but a lag pair has no physical meaning "
                "without it. Set it to 1000.0 for mdCATH (see docs/open_questions.md "
                "section 2)."
            )
        # fail at construction, not at the first pair
        for lag in self.lags_ps:
            exact_lag_frames(lag, self.mdcath.ps_per_frame)

    @property
    def ps_per_frame(self) -> float:
        return float(self.mdcath.ps_per_frame)

    @property
    def past_frames(self) -> int:
        """History frames *before* the current one: 1 by default, 0 if length 1."""
        return self.history_length - 1

    def lag_frames(self) -> tuple[int, ...]:
        return tuple(exact_lag_frames(lag, self.ps_per_frame) for lag in self.lags_ps)

    def as_dict(self) -> dict:
        """JSON-serialisable record, for the manifest metadata."""
        md = {
            k: (str(v) if k == "dtype" else v)
            for k, v in self.mdcath.__dict__.items()
        }
        md["frames_per_trajectory"] = "IGNORED BY LAG PAIRS"
        return {
            "mdcath": md,
            "lags_ps": list(self.lags_ps),
            "lag_frames": list(self.lag_frames()),
            "history_length": self.history_length,
            "require_current_force_labels": self.require_current_force_labels,
            "pairs_per_trajectory": self.pairs_per_trajectory,
            "max_trajectories_per_domain": self.max_trajectories_per_domain,
            "max_domains": self.max_domains,
            "max_pairs": self.max_pairs,
            "selection": self.selection,
            "seed": self.seed,
        }


@dataclass(frozen=True)
class LagPair:
    """One transition, addressed by raw frame numbers.

    Args:
        domain: CATH domain id.
        temperature: mdCATH temperature group name, e.g. ``"320"``.
        replica: mdCATH replica group name, e.g. ``"0"``.
        current_frame: ``t``, the frame the model conditions on.
        lag_frames: ``t+lag_frames`` is the target frame.
        lag_ps: the same lag as a physical time.
        past_frames: how many frames before ``t`` are provided as history.
    """

    domain: str
    temperature: str
    replica: str
    current_frame: int
    lag_frames: int
    lag_ps: float
    past_frames: int = 1

    @property
    def future_frame(self) -> int:
        return self.current_frame + self.lag_frames

    @property
    def history_frames(self) -> tuple[int, ...]:
        """Frames before ``t``, oldest first. Empty when ``history_length == 1``."""
        return tuple(range(self.current_frame - self.past_frames, self.current_frame))

    @property
    def all_frames(self) -> tuple[int, ...]:
        return (*self.history_frames, self.current_frame, self.future_frame)

    @property
    def trajectory(self) -> str:
        return f"{self.temperature}/{self.replica}"

    @property
    def pair_id(self) -> str:
        """Stable identity: domain, temperature, replica, both frames and the lag."""
        return (
            f"{self.domain}/{self.temperature}/{self.replica}"
            f"/t{self.current_frame}/f{self.future_frame}/lag{self.lag_ps:g}ps"
        )

    @property
    def lag_ns(self) -> float:
        return self.lag_ps / 1000.0

    def as_row(self) -> list:
        return [
            self.domain, self.temperature, self.replica,
            self.current_frame, self.lag_frames, self.lag_ps, self.past_frames,
        ]

    @staticmethod
    def from_row(row: Sequence) -> "LagPair":
        return LagPair(
            domain=str(row[0]), temperature=str(row[1]), replica=str(row[2]),
            current_frame=int(row[3]), lag_frames=int(row[4]), lag_ps=float(row[5]),
            past_frames=int(row[6]),
        )


@dataclass
class LagPairManifest:
    """The enumerated pairs plus everything needed to reproduce them."""

    pairs: tuple[LagPair, ...]
    metadata: dict

    def __len__(self) -> int:
        return len(self.pairs)

    @property
    def domains(self) -> list[str]:
        return sorted({p.domain for p in self.pairs})

    @property
    def trajectories(self) -> set[tuple[str, str, str]]:
        return {(p.domain, p.temperature, p.replica) for p in self.pairs}

    def counts_by_lag(self) -> dict[float, int]:
        out: dict[float, int] = {}
        for p in self.pairs:
            out[p.lag_ps] = out.get(p.lag_ps, 0) + 1
        return dict(sorted(out.items()))

    def content_hash(self) -> str:
        """SHA-256 over the pair rows alone -- the identity of the *index*."""
        digest = hashlib.sha256()
        for pair in self.pairs:
            digest.update(json.dumps(pair.as_row(), sort_keys=True).encode())
        return digest.hexdigest()

    def to_payload(self) -> dict:
        meta = dict(self.metadata)
        meta["content_hash"] = self.content_hash()
        meta["num_pairs"] = len(self.pairs)
        return {
            "version": MANIFEST_VERSION,
            "columns": ["domain", "temperature", "replica", "current_frame",
                        "lag_frames", "lag_ps", "past_frames"],
            "metadata": meta,
            "pairs": [p.as_row() for p in self.pairs],
        }

    def save(self, path: str) -> str:
        """Write atomically and return the content hash.

        Atomic because a manifest is what a run's reproducibility rests on, and a
        half-written one is worse than none: it parses.
        """
        payload = self.to_payload()
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        tmp = f"{path}.tmp"
        with open(tmp, "w") as fh:
            json.dump(payload, fh, indent=1, sort_keys=False)
        os.replace(tmp, path)
        return payload["metadata"]["content_hash"]

    @staticmethod
    def load(path: str) -> "LagPairManifest":
        payload = json.load(open(path))
        version = payload.get("version")
        if version != MANIFEST_VERSION:
            raise ValueError(
                f"{path}: manifest version {version!r}, this code writes "
                f"{MANIFEST_VERSION}. Rebuild it rather than reinterpreting old "
                "columns under new meanings."
            )
        manifest = LagPairManifest(
            pairs=tuple(LagPair.from_row(row) for row in payload["pairs"]),
            metadata=payload["metadata"],
        )
        stored = payload["metadata"].get("content_hash")
        if stored is not None and stored != manifest.content_hash():
            raise ValueError(
                f"{path}: content hash mismatch (stored {stored[:12]}, recomputed "
                f"{manifest.content_hash()[:12]}). The file was edited or truncated."
            )
        return manifest

    def assert_disjoint(self, other: "LagPairManifest", *, names=("train", "val")) -> None:
        """Refuse a domain that appears in both splits."""
        overlap = sorted(set(self.domains) & set(other.domains))
        if overlap:
            raise ValueError(
                f"{len(overlap)} domain(s) appear in both {names[0]} and {names[1]}: "
                f"{overlap[:5]}. Consecutive frames of one protein are highly "
                "correlated, so this would measure memorisation."
            )


# --------------------------------------------------------------------------
# split restoration
# --------------------------------------------------------------------------


def load_phase1_split(snapshot_path: str) -> tuple[list[str], list[str]]:
    """Read the domain split recorded by the Phase 1 run."""
    payload = json.load(open(snapshot_path))
    if "split" not in payload:
        raise ValueError(
            f"{snapshot_path} has no 'split' section; it is not a Phase 1 config "
            "snapshot written by scripts/train_phase1.py"
        )
    split = payload["split"]
    return list(split["train_domains"]), list(split["val_domains"])


def restore_phase1_split(
    config: MdCathConfig,
    *,
    snapshot_path: Optional[str] = None,
    val_fraction: float = 0.2,
    split_seed: int = 0,
) -> tuple[list[str], list[str]]:
    """Recover Phase 1's train/validation domain lists.

    Two independent routes, cross-checked when both are available:

    1. recompute ``split_domains(eligible_domains, val_fraction, split_seed)``,
       where *eligible* applies the same ``max_residues`` filter Phase 1 used;
    2. read the lists recorded in ``runs/*/config_snapshot.json``.

    They are required to agree. The snapshot is the record of what actually ran;
    the recomputation proves the recipe still reproduces it, which is what makes
    the split auditable rather than a file someone has to trust.
    """
    eligible = _eligible_domains(config)
    train, val = split_domains(eligible, val_fraction, split_seed)
    if snapshot_path and os.path.exists(snapshot_path):
        snap_train, snap_val = load_phase1_split(snapshot_path)
        if snap_train != train or snap_val != val:
            raise ValueError(
                f"the recomputed Phase 1 split does not match {snapshot_path}: "
                f"{len(train)}/{len(val)} recomputed vs {len(snap_train)}/"
                f"{len(snap_val)} recorded, first differing domain "
                f"{_first_difference(train, snap_train) or _first_difference(val, snap_val)!r}. "
                "Phase 1.5 must train and validate on Phase 1's domains; refusing "
                "to guess which list is right."
            )
    overlap = set(train) & set(val)
    if overlap:
        raise ValueError(f"domain leak between splits: {sorted(overlap)[:5]}")
    return train, val


def _first_difference(a: Sequence[str], b: Sequence[str]) -> Optional[str]:
    for x, y in zip(a, b):
        if x != y:
            return f"{x} != {y}"
    return None if len(a) == len(b) else f"length {len(a)} != {len(b)}"


def _eligible_domains(config: MdCathConfig) -> list[str]:
    """Domains that pass Phase 1's ``max_residues`` filter, from attributes only."""
    import h5py  # noqa: PLC0415 - optional dependency
    import glob  # noqa: PLC0415

    out = []
    for path in sorted(glob.glob(os.path.join(config.data_dir, "*.h5"))):
        domain = os.path.basename(path)[len("mdcath_dataset_") : -len(".h5")]
        if config.max_residues is None:
            out.append(domain)
            continue
        with h5py.File(path, "r") as handle:
            if int(handle[domain].attrs["numResidues"]) <= config.max_residues:
                out.append(domain)
    return out


# --------------------------------------------------------------------------
# manifest construction
# --------------------------------------------------------------------------


def _select(candidates: np.ndarray, k: Optional[int], selection: str, rng) -> np.ndarray:
    """Deterministically thin ``candidates`` to at most ``k`` entries."""
    if k is None or len(candidates) <= k:
        return candidates
    if selection == "random":
        picks = np.sort(rng.choice(len(candidates), size=k, replace=False))
        return candidates[picks]
    # Even stride: nested across sizes, and it spans the trajectory rather than
    # taking its first k frames, which would sample one corner of the dynamics.
    slots = np.unique(np.linspace(0, len(candidates) - 1, k).round().astype(int))
    return candidates[slots]


def _sha256_of(path: Optional[str]) -> Optional[str]:
    if not path or not os.path.exists(path):
        return None
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_lag_pair_manifest(
    config: LagPairConfig,
    domains: Sequence[str],
    *,
    split: str,
    reader: Optional[MdCathDataset] = None,
) -> LagPairManifest:
    """Enumerate every admissible lag pair over ``domains``.

    Reads ``numFrames`` attributes only -- no coordinate is touched here, so a
    manifest over the full 623 GB corpus costs seconds and no memory.

    A candidate ``t`` survives when, for its lag:

    * ``t - past_frames >= 0`` and ``t + lag_frames <= n_frames - 1``;
    * none of its frames is coordinate-quarantined;
    * its trajectory has usable force labels, unless
      ``require_current_force_labels`` is False.
    """
    owns_reader = reader is None
    if reader is None:
        reader = MdCathDataset(config.mdcath, domains=list(domains), build_index=False)
    try:
        rng = np.random.default_rng(config.seed)
        lag_frames = config.lag_frames()
        past = config.past_frames
        pairs: list[LagPair] = []
        n_traj = 0
        skipped_force = 0

        kept_domains = 0
        for domain in sorted(domains):
            if config.max_domains is not None and kept_domains >= config.max_domains:
                break
            group = reader._open(domain)[domain]
            if (
                config.mdcath.max_residues is not None
                and int(group.attrs["numResidues"]) > config.mdcath.max_residues
            ):
                continue
            temps = sorted(k for k in group.keys() if k.isdigit())
            if config.mdcath.temperatures is not None:
                wanted = {str(t) for t in config.mdcath.temperatures}
                temps = [t for t in temps if t in wanted]

            domain_pairs = 0
            traj_used = 0
            for temp in temps:
                reps = sorted(group[temp].keys())
                if config.mdcath.replicas is not None:
                    wanted_r = {str(r) for r in config.mdcath.replicas}
                    reps = [r for r in reps if r in wanted_r]
                for rep in reps:
                    if (
                        config.max_trajectories_per_domain is not None
                        and traj_used >= config.max_trajectories_per_domain
                    ):
                        break
                    n_frames = int(group[temp][rep].attrs["numFrames"])
                    if n_frames == 0:
                        continue
                    force_ok = f"{temp}/{rep}" not in reader.quarantine.get(domain, set())
                    if config.require_current_force_labels and not force_ok:
                        skipped_force += 1
                        continue
                    corrupt = reader.coord_quarantine.get(domain, {}).get(f"{temp}/{rep}", set())

                    traj_had_pair = False
                    for lag_ps, lag in zip(config.lags_ps, lag_frames):
                        first = past
                        last = n_frames - 1 - lag  # inclusive
                        if last < first:
                            continue  # trajectory too short for this lag
                        starts = np.arange(first, last + 1)
                        if corrupt:
                            bad = np.asarray(sorted(corrupt))
                            keep = np.ones(len(starts), dtype=bool)
                            for offset in range(-past, 1):
                                keep &= ~np.isin(starts + offset, bad)
                            keep &= ~np.isin(starts + lag, bad)
                            starts = starts[keep]
                        starts = _select(starts, config.pairs_per_trajectory,
                                         config.selection, rng)
                        for t in starts:
                            pairs.append(
                                LagPair(domain=domain, temperature=temp, replica=rep,
                                        current_frame=int(t), lag_frames=lag,
                                        lag_ps=float(lag_ps), past_frames=past)
                            )
                            domain_pairs += 1
                            traj_had_pair = True
                    if traj_had_pair:
                        n_traj += 1
                        traj_used += 1
            if domain_pairs:
                kept_domains += 1

        if config.max_pairs is not None and len(pairs) > config.max_pairs:
            # Even stride over the whole list, so every domain keeps a share
            # instead of the first few domains taking the whole budget.
            slots = np.unique(
                np.linspace(0, len(pairs) - 1, config.max_pairs).round().astype(int)
            )
            pairs = [pairs[i] for i in slots]

        metadata = {
            "created_by": "force_md.data.adapters.lag_pairs.build_lag_pair_manifest",
            "split": split,
            "config": config.as_dict(),
            "ps_per_frame": config.ps_per_frame,
            "lags_ps": list(config.lags_ps),
            "lag_frames": list(lag_frames),
            "history_length": config.history_length,
            "seed": config.seed,
            "selection": config.selection,
            "domains": sorted({p.domain for p in pairs}),
            "num_domains": len({p.domain for p in pairs}),
            "num_trajectories": len({(p.domain, p.temperature, p.replica) for p in pairs}),
            "num_pairs_by_lag": {str(k): v for k, v in
                                 LagPairManifest(tuple(pairs), {}).counts_by_lag().items()},
            "trajectories_skipped_for_force_quarantine": skipped_force,
            "quarantine": {
                "force_path": config.mdcath.quarantine_path,
                "force_sha256": _sha256_of(config.mdcath.quarantine_path),
                "coord_path": config.mdcath.coord_quarantine_path,
                "coord_sha256": _sha256_of(config.mdcath.coord_quarantine_path),
            },
        }
        return LagPairManifest(pairs=tuple(pairs), metadata=metadata)
    finally:
        if owns_reader:
            reader.close()


# --------------------------------------------------------------------------
# dataset
# --------------------------------------------------------------------------


@dataclass
class LagPairExample:
    """One transition.

    Args:
        pair: which transition this is.
        current: the Phase 1 state at ``t``, with its force labels. The labels
            are for the oracle arm and for Phase 1 supervision; the model's
            forward pass does not read them.
        history: frames before ``t``, oldest first. Geometry only.
        future: the frame at ``t + lag``. **A label.** It is a
            :class:`FrameGeometry`, not a state, so it cannot be handed to the
            encoder by accident.
    """

    pair: LagPair
    current: TrainingExample
    history: tuple[FrameGeometry, ...]
    future: FrameGeometry


@dataclass
class LagPairBatch:
    """A collated batch of transitions."""

    pairs: tuple[LagPair, ...]
    current: HierarchicalProteinBatch
    hidden_force_target: Optional[Tensor]
    history: tuple[FrameGeometry, ...]
    future: FrameGeometry
    lag_ps: Tensor
    lag_frames: Tensor

    @property
    def num_graphs(self) -> int:
        return self.current.num_graphs

    def to(self, device, non_blocking: bool = False) -> "LagPairBatch":
        return replace(
            self,
            current=self.current.to(device, non_blocking),
            hidden_force_target=None if self.hidden_force_target is None
            else self.hidden_force_target.to(device, non_blocking=non_blocking),
            history=tuple(h.to(device, non_blocking) for h in self.history),
            future=self.future.to(device, non_blocking),
            lag_ps=self.lag_ps.to(device, non_blocking=non_blocking),
            lag_frames=self.lag_frames.to(device, non_blocking=non_blocking),
        )

    def validate(self) -> None:
        self.current.validate()
        for frame in self.history:
            frame.matches(self.current)
        self.future.matches(self.current)


def collate_lag_pairs(examples: Sequence[LagPairExample]) -> LagPairBatch:
    """Collate transitions into one ragged batch.

    Every example must carry the same number of history frames; a batch with
    mixed history lengths would silently give some proteins more context than
    others, which is exactly the confound the ablation exists to avoid.
    """
    if not examples:
        raise ValueError("cannot collate an empty list of lag pairs")
    lengths = {len(e.history) for e in examples}
    if len(lengths) != 1:
        raise ValueError(f"examples disagree on history length: {sorted(lengths)}")
    n_history = lengths.pop()

    current = collate_batches([e.current.batch for e in examples], validate=False)
    hidden = [e.current.hidden_force_target for e in examples]
    hidden_target = None if any(h is None for h in hidden) else torch.cat(hidden, dim=0)
    history = tuple(
        collate_frame_geometries([e.history[k] for e in examples])
        for k in range(n_history)
    )
    future = collate_frame_geometries([e.future for e in examples])
    dtype = current.atoms.positions.dtype
    return LagPairBatch(
        pairs=tuple(e.pair for e in examples),
        current=current,
        hidden_force_target=hidden_target,
        history=history,
        future=future,
        lag_ps=torch.tensor([e.pair.lag_ps for e in examples], dtype=dtype),
        lag_frames=torch.tensor([e.pair.lag_frames for e in examples], dtype=torch.int64),
    )


class LagPairDataset(torch.utils.data.Dataset):
    """One item = one ``(history, current) -> future`` transition.

    Composes :class:`MdCathDataset` rather than subclassing it: the frame
    readers, both quarantines, the saturation/collapse backstops, the per-frame
    centring and the PBC check are reused unchanged, while the index is this
    class's own.
    """

    def __init__(self, config: LagPairConfig, manifest: LagPairManifest):
        self.config = config
        self.manifest = manifest
        self.pairs = manifest.pairs
        self._check_manifest_matches_config()
        self.frames = MdCathDataset(
            config.mdcath, domains=manifest.domains or None, build_index=False
        )
        # Close before any fork, exactly as Phase 1 does: an inherited h5py
        # handle is not safe to use from a DataLoader worker.
        self.frames.close()

    def _check_manifest_matches_config(self) -> None:
        """A manifest built under different rules is not this dataset's index."""
        meta = self.manifest.metadata
        if not meta:
            return
        if meta.get("history_length") not in (None, self.config.history_length):
            raise ValueError(
                f"manifest was built with history_length="
                f"{meta['history_length']} but the config asks for "
                f"{self.config.history_length}. The pair rows encode the history "
                "frames, so mixing the two would read frames the index never checked."
            )
        if meta.get("ps_per_frame") not in (None, self.config.ps_per_frame):
            raise ValueError(
                f"manifest ps_per_frame {meta['ps_per_frame']} != config "
                f"{self.config.ps_per_frame}; every lag in the manifest would be "
                "relabelled with a different physical time."
            )

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, i: int) -> LagPairExample:
        pair = self.pairs[i]
        if pair.past_frames != self.config.past_frames:
            raise ValueError(
                f"{pair.pair_id}: manifest row carries past_frames="
                f"{pair.past_frames}, config says {self.config.past_frames}"
            )
        topo = self.frames.topology_for(pair.domain)

        coords, forces, forces_valid = self.frames.load_frame_arrays(
            pair.domain, pair.temperature, pair.replica, pair.current_frame
        )
        current = self.frames.build_example(
            pair.domain, pair.temperature, pair.replica, pair.current_frame,
            coords, forces, forces_valid,
        )
        history = tuple(
            self._geometry(topo, pair, frame) for frame in pair.history_frames
        )
        future = self._geometry(topo, pair, pair.future_frame)
        return LagPairExample(pair=pair, current=current, history=history, future=future)

    def _geometry(self, topo: _Topology, pair: LagPair, frame: int) -> FrameGeometry:
        """Geometry of one non-current frame, through the same safety checks."""
        coords, _, _ = self.frames.load_frame_arrays(
            pair.domain, pair.temperature, pair.replica, frame
        )
        dtype = self.config.mdcath.dtype
        keep = np.nonzero(topo.represented)[0]
        safe = lambda idx: np.where(idx >= 0, idx, 0)  # noqa: E731
        tensor = lambda a, d: torch.tensor(a, dtype=d)  # noqa: E731
        n_res = topo.num_residues
        return FrameGeometry(
            positions=tensor(coords[keep], dtype),
            n_positions=tensor(coords[safe(topo.n_index)], dtype),
            ca_positions=tensor(coords[safe(topo.ca_index)], dtype),
            c_positions=tensor(coords[safe(topo.c_index)], dtype),
            frame_valid=tensor(topo.frame_atoms_complete, torch.bool),
            atom_batch_index=torch.zeros(len(keep), dtype=torch.int64),
            residue_batch_index=torch.zeros(n_res, dtype=torch.int64),
            frame_index=torch.tensor([frame], dtype=torch.int64),
        )

    def close(self) -> None:
        self.frames.close()

    @property
    def domains(self) -> list[str]:
        return self.manifest.domains
