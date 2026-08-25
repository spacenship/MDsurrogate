"""Frozen Phase 1 as a feature extractor for the transition probe.

Phase 1 is finished and is not retrained here. It is loaded from its checkpoint,
put in ``eval()``, frozen, and run under ``no_grad`` to produce a structured
bundle that the Checkpoint 4 conditioners consume.

**The type system carries the one guarantee that matters.** Ground-truth forces
are labels. If they ever reach a production arm's conditioning path, every
Phase 1.5 number becomes meaningless -- the model would be reading the answer's
neighbourhood -- and nothing in the loss curve would look wrong. So the GT forces
are not a field of :class:`FeatureBundle` that happens to be ``None`` in
production; they live in a **different class**, :class:`OracleFeatureBundle`,
produced by a **different method**, and a production conditioner that asks for
them fails at ``isinstance`` rather than silently succeeding.

That is also why ``FeatureBundle`` carries no ``forces`` attribute at all: a
``None`` field invites ``if bundle.forces is not None``, and one day it will not
be None.

**What "frozen" is checked to mean.** Not just ``requires_grad=False``: the test
runs a downstream head, backpropagates a loss through the bundle, and requires
every Phase 1 parameter to still have ``grad is None``.

**Contract, not assumption.** The checkpoint records what Phase 1 promised --
``physics_latent_irreps``, ``physics_latent_dim``, ``row_order``, ``frame``,
``target_scope``, ``num_cycles``, ``lmax``. This module rebuilds the model, asks
it for its contract again, and refuses to run if the two disagree or if a
runtime expectation is violated. A checkpoint whose contract cannot be verified
is more dangerous than one that fails to load: it produces plausible features of
the wrong shape or the wrong frame.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
from dataclasses import dataclass, replace
from typing import Any, Iterable, Optional, Sequence

import torch
from torch import Tensor, nn

from ..data.contracts import HierarchicalProteinBatch
from ..geometry.frames import (
    ResidueFrames,
    atom_local_coordinates,
    link_backbone_to_atom_positions,
)
from ..models.local_physics import LocalPhysicsConfig, LocalPhysicsModel

__all__ = [
    "FeatureBundle",
    "OracleFeatureBundle",
    "FrozenPhase1Extractor",
    "Phase1FeatureCache",
    "assert_production_safe",
    "checkpoint_fingerprint",
    "split_bundle",
    "merge_bundles",
]

#: Bumped when the bundle's field set changes, so a stale cache shard is rejected
#: instead of being read under new column meanings.
FEATURE_FORMAT_VERSION = 1


@dataclass
class FeatureBundle:
    """Everything a **production** conditioner may see. No ground truth here.

    Row orders are the batch's own: residue rows align with
    ``batch.residues`` and atom rows with ``batch.atoms``, so an index means the
    same thing in the bundle, the batch and the transition target.

    Args:
        physics_latent: ``[N_res, D]`` Phase 1's residue representation, in the
            **global frame**, carrying ``physics_latent_irreps``. This is not new
            observational information -- it is ``g(q_t, sequence, temperature)``,
            a representation pretrained with force supervision.
        physics_latent_irreps / latent_row_order / latent_frame: the contract
            those numbers were produced under, carried alongside them so a
            consumer can assert rather than assume.
        atom_force_mean: ``[N_atom, 3]`` **predicted** atomic force, global frame.
        atom_force_logvar: ``[N_atom, 3]`` predicted diagonal log-variance, in the
            **residue-local** frame (a diagonal covariance is only meaningful in a
            fixed frame).
        residue_force_mean / residue_torque_mean: ``[N_res, 3]`` predicted, global
            frame; the torque is about ``residue_torque_origin`` (the CA).
        residue_force_logvar / residue_torque_logvar: ``[N_res, 3]`` local frame.
        atom_to_residue: ``[N_atom]`` parent residue of each atom.
        atom_local_coordinates: ``[N_atom, 3]`` ``y_ia = R_i^T (x_ia - r_i)``.
        frames: the current residue frames ``(R_i, r_i)``.
        atom_is_heavy / atom_is_backbone: ``[N_atom]`` bool, chemistry flags the
            conditioner may key on.
        atomic_number / atom_name_id: ``[N_atom]`` int64. Chemistry, not geometry
            and not a label, so it is production-safe. These are what the atom-set
            pooling embeds instead of an invented van der Waals radius table: the
            element and the CHARMM atom name are recorded in the file, a radius
            would be a constant someone chose.
        residue_valid: ``[N_res]`` bool -- frame usable and residue not masked.
        atom_valid: ``[N_atom]`` bool -- parent residue valid.
        residue_batch_index / atom_batch_index: ``[N]`` graph ids.
        num_graphs: graphs in this bundle.
    """

    physics_latent: Tensor
    physics_latent_irreps: str
    latent_row_order: str
    latent_frame: str
    atom_force_mean: Tensor
    atom_force_logvar: Tensor
    residue_force_mean: Tensor
    residue_torque_mean: Tensor
    residue_torque_origin: Tensor
    residue_force_logvar: Tensor
    residue_torque_logvar: Tensor
    atom_to_residue: Tensor
    atom_local_coordinates: Tensor
    frames: ResidueFrames
    atom_is_heavy: Tensor
    atom_is_backbone: Tensor
    atomic_number: Tensor
    atom_name_id: Tensor
    residue_valid: Tensor
    atom_valid: Tensor
    residue_batch_index: Tensor
    atom_batch_index: Tensor
    num_graphs: int

    @property
    def num_residues(self) -> int:
        return int(self.physics_latent.shape[0])

    @property
    def num_atoms(self) -> int:
        return int(self.atom_force_mean.shape[0])

    @property
    def latent_dim(self) -> int:
        return int(self.physics_latent.shape[1])

    def to(self, device) -> "FeatureBundle":
        moved = {
            f.name: getattr(self, f.name).to(device)
            for f in dataclasses.fields(self)
            if isinstance(getattr(self, f.name), Tensor)
        }
        return replace(self, frames=self.frames.to(device), **moved)

    def requires_grad_state(self) -> dict[str, bool]:
        """Which tensors carry gradient. Every one must be False when frozen."""
        return {
            f.name: bool(getattr(self, f.name).requires_grad)
            for f in dataclasses.fields(self)
            if isinstance(getattr(self, f.name), Tensor)
        }


@dataclass
class OracleFeatureBundle:
    """Ground-truth atomic forces. **Diagnostic arm only.**

    Deliberately a separate class from :class:`FeatureBundle` and deliberately
    not a superset of it: an oracle bundle cannot be passed where a production
    bundle is expected, and vice versa, so the separation is enforced by the type
    checker and by ``isinstance``, not by a naming convention.

    The oracle arm exists to answer one question -- is instantaneous force useful
    for a 1-4 ns transition *at all*, independent of how well Phase 1 predicts it?
    A result where the oracle also fails to beat the structure-only baseline says
    something quite different from one where only the predicted-force arms fail.

    Args:
        atom_force: ``[N_atom, 3]`` the mdCATH force label, global frame.
        atom_force_valid: ``[N_atom]`` bool; False where the label is quarantined.
        production: the production bundle for the same batch, so an oracle arm
            still has the structural features.
    """

    atom_force: Tensor
    atom_force_valid: Tensor
    production: FeatureBundle

    def to(self, device) -> "OracleFeatureBundle":
        return OracleFeatureBundle(
            atom_force=self.atom_force.to(device),
            atom_force_valid=self.atom_force_valid.to(device),
            production=self.production.to(device),
        )


def assert_production_safe(bundle: Any) -> FeatureBundle:
    """Return ``bundle`` if it is production-safe, else raise.

    Called by every production conditioner on the way in. The error names the
    problem in full because the failure it prevents -- a ground-truth force
    reaching a production arm -- is invisible once it happens.
    """
    if isinstance(bundle, OracleFeatureBundle):
        raise TypeError(
            "an OracleFeatureBundle reached a production conditioner. Ground-truth "
            "forces are labels, never inputs: an arm that reads them is not "
            "measuring what Phase 1.5 claims to measure. Use the oracle_force arm "
            "explicitly, or pass bundle.production."
        )
    if not isinstance(bundle, FeatureBundle):
        raise TypeError(f"expected a FeatureBundle, got {type(bundle).__name__}")
    return bundle


def checkpoint_fingerprint(path: str) -> str:
    """SHA-256 of the checkpoint file, for cache keys and run records."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 22), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _config_fingerprint(config: LocalPhysicsConfig) -> str:
    payload = json.dumps(dataclasses.asdict(config), sort_keys=True, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()


class FrozenPhase1Extractor(nn.Module):
    """Phase 1, loaded from its checkpoint and used read-only.

    Args:
        model: the restored :class:`LocalPhysicsModel`.
        contract: the ``latent_contract`` recorded in the checkpoint.
        freeze: eval mode, ``requires_grad=False`` on every parameter, forward
            under ``no_grad``. Leaving it True is the Phase 1.5 default and the
            only configuration the ablation is run in; False exists so a later
            joint fine-tune is a config change rather than a new class.
        metadata: provenance (checkpoint path, hashes, step) written into run
            records so a result can be traced to the weights that produced it.
    """

    def __init__(
        self,
        model: LocalPhysicsModel,
        contract: dict,
        *,
        freeze: bool = True,
        metadata: Optional[dict] = None,
    ):
        super().__init__()
        self.phase1 = model
        self.contract = dict(contract)
        self.freeze = freeze
        self.metadata = dict(metadata or {})
        if freeze:
            self.phase1.eval()
            for parameter in self.phase1.parameters():
                parameter.requires_grad_(False)

    # -- construction ------------------------------------------------------

    @classmethod
    def from_checkpoint(
        cls,
        path: str,
        *,
        device: str | torch.device = "cpu",
        freeze: bool = True,
        expect: Optional[dict] = None,
        compute_fingerprint: bool = True,
    ) -> "FrozenPhase1Extractor":
        """Restore Phase 1 without touching the file.

        The checkpoint is opened read-only and nothing is written back -- no
        optimiser state, no re-save. Phase 1 is a finished artefact and Phase 1.5
        must be re-runnable against the identical weights.

        Args:
            path: the Phase 1 checkpoint, e.g. ``runs/phase1_full/last.pt``.
            expect: optional subset of the latent contract that the *runtime*
                config requires (e.g. ``{"physics_latent_dim": 152}``). Every key
                present is checked; a mismatch raises.

        Raises:
            ValueError: if the rebuilt model's contract disagrees with the
                recorded one, or with ``expect``.
        """
        payload = torch.load(path, map_location=device, weights_only=False)
        for key in ("state_dict", "model_config"):
            if key not in payload:
                raise ValueError(
                    f"{path} has no {key!r}; it is not a Phase 1 checkpoint written "
                    "by Phase1Trainer.save_checkpoint"
                )
        model_config: LocalPhysicsConfig = payload["model_config"]
        model = LocalPhysicsModel(model_config)
        model.load_state_dict(payload["state_dict"])
        model.to(device)

        recorded = payload.get("latent_contract")
        live = model.latent_contract()
        if recorded is not None:
            _check_contract(recorded, live, source=f"{path} (recorded)")
        if expect:
            _check_contract(expect, live, source="runtime config", subset=True)

        metadata = {
            "checkpoint_path": os.path.abspath(path),
            "step": payload.get("step"),
            "config_hash": _config_fingerprint(model_config),
            "latent_contract": live,
            "normalizer": (
                dataclasses.asdict(payload["normalizer"])
                if dataclasses.is_dataclass(payload.get("normalizer"))
                else payload.get("normalizer")
            ),
        }
        if compute_fingerprint:
            metadata["checkpoint_sha256"] = checkpoint_fingerprint(path)
        return cls(model, live, freeze=freeze, metadata=metadata)

    # -- extraction --------------------------------------------------------

    def forward(self, batch: HierarchicalProteinBatch) -> FeatureBundle:
        """Run Phase 1 and package what a production conditioner may use."""
        if self.freeze:
            with torch.no_grad():
                return self._extract(batch)
        return self._extract(batch)

    def _extract(self, batch: HierarchicalProteinBatch) -> FeatureBundle:
        linked = link_backbone_to_atom_positions(batch)
        output = self.phase1(linked)
        local, frames = atom_local_coordinates(linked)

        residue_valid = frames.valid & linked.residues.mask
        atom_valid = residue_valid[linked.atoms.atom_to_residue]

        detach = (lambda t: t.detach()) if self.freeze else (lambda t: t)
        bundle = FeatureBundle(
            physics_latent=detach(output.physics_latent),
            physics_latent_irreps=output.physics_latent_irreps,
            latent_row_order=str(self.contract.get("row_order", "")),
            latent_frame=str(self.contract.get("frame", "global")),
            atom_force_mean=detach(output.atom_force_mean),
            atom_force_logvar=detach(output.atom_force_logvar),
            residue_force_mean=detach(output.residue_force_mean),
            residue_torque_mean=detach(output.residue_torque_mean),
            residue_torque_origin=detach(output.residue_torque_origin),
            residue_force_logvar=detach(output.residue_force_logvar),
            residue_torque_logvar=detach(output.residue_torque_logvar),
            atom_to_residue=linked.atoms.atom_to_residue,
            atom_local_coordinates=detach(local),
            frames=frames,
            atom_is_heavy=linked.atoms.is_heavy,
            atom_is_backbone=linked.atoms.is_backbone,
            atomic_number=linked.atoms.atomic_number,
            atom_name_id=linked.atoms.atom_name_id,
            residue_valid=residue_valid,
            atom_valid=atom_valid,
            residue_batch_index=linked.residues.batch_index,
            atom_batch_index=linked.atoms.batch_index,
            num_graphs=linked.num_graphs,
        )
        _check_bundle_against_contract(bundle, self.contract)
        return bundle

    def oracle_bundle(self, batch: HierarchicalProteinBatch) -> OracleFeatureBundle:
        """Ground-truth forces plus the production features. Diagnostic arm only.

        Raises:
            ValueError: if the batch carries no force labels. The oracle arm has
                no meaning without them and must not silently fall back to the
                predicted forces, which would make arm E a duplicate of arm D.
        """
        if batch.atoms.forces is None:
            raise ValueError(
                "the oracle arm needs batch.atoms.forces and this batch has none. "
                "Falling back to predicted forces would silently turn the oracle "
                "arm into a copy of the production one."
            )
        valid = (
            batch.atoms.force_valid
            if batch.atoms.force_valid is not None
            else torch.ones_like(batch.atoms.is_heavy)
        )
        return OracleFeatureBundle(
            atom_force=batch.atoms.forces.detach(),
            atom_force_valid=valid,
            production=self.forward(batch),
        )


def _check_contract(expected: dict, live: dict, *, source: str, subset: bool = False) -> None:
    differences = []
    keys = expected.keys() if subset else set(expected) | set(live)
    for key in sorted(keys):
        if key not in expected or key not in live:
            if not subset:
                differences.append(f"{key}: {expected.get(key)!r} vs {live.get(key)!r}")
            continue
        if expected[key] != live[key]:
            differences.append(f"{key}: expected {expected[key]!r}, model gives {live[key]!r}")
    if differences:
        raise ValueError(
            f"Phase 1 latent contract mismatch against {source}:\n  "
            + "\n  ".join(differences)
            + "\nThe features would have the wrong width, row order or frame, and "
            "nothing downstream would notice. Refusing to run."
        )


def _check_bundle_against_contract(bundle: FeatureBundle, contract: dict) -> None:
    expected_dim = contract.get("physics_latent_dim")
    if expected_dim is not None and bundle.latent_dim != int(expected_dim):
        raise ValueError(
            f"physics_latent has width {bundle.latent_dim} but the contract "
            f"promises {expected_dim}"
        )
    expected_irreps = contract.get("physics_latent_irreps")
    if expected_irreps is not None and bundle.physics_latent_irreps != expected_irreps:
        raise ValueError(
            f"physics_latent carries irreps {bundle.physics_latent_irreps!r} but the "
            f"contract promises {expected_irreps!r}"
        )
    if bundle.physics_latent.shape[0] != bundle.residue_batch_index.shape[0]:
        raise ValueError(
            f"physics_latent has {bundle.physics_latent.shape[0]} rows for "
            f"{bundle.residue_batch_index.shape[0]} residues; the row order contract "
            "('aligned with batch.residues') is broken"
        )


# --------------------------------------------------------------------------
# per-graph split / merge, for caching
# --------------------------------------------------------------------------

_RESIDUE_FIELDS = (
    "physics_latent", "residue_force_mean", "residue_torque_mean",
    "residue_torque_origin", "residue_force_logvar", "residue_torque_logvar",
    "residue_valid",
)
_ATOM_FIELDS = (
    "atom_force_mean", "atom_force_logvar", "atom_local_coordinates",
    "atom_is_heavy", "atom_is_backbone", "atomic_number", "atom_name_id",
    "atom_valid",
)


def split_bundle(bundle: FeatureBundle, graph: int) -> FeatureBundle:
    """One graph's rows, renumbered as a standalone single-graph bundle."""
    residues = (bundle.residue_batch_index == graph).nonzero(as_tuple=True)[0]
    atoms = (bundle.atom_batch_index == graph).nonzero(as_tuple=True)[0]
    remap = torch.full_like(bundle.residue_batch_index, -1)
    remap[residues] = torch.arange(residues.numel(), device=residues.device)

    fields = {name: getattr(bundle, name)[residues] for name in _RESIDUE_FIELDS}
    fields.update({name: getattr(bundle, name)[atoms] for name in _ATOM_FIELDS})
    return replace(
        bundle,
        **fields,
        atom_to_residue=remap[bundle.atom_to_residue[atoms]],
        frames=ResidueFrames(
            rotation=bundle.frames.rotation[residues],
            origin=bundle.frames.origin[residues],
            valid=bundle.frames.valid[residues],
        ),
        residue_batch_index=torch.zeros_like(residues),
        atom_batch_index=torch.zeros_like(atoms),
        num_graphs=1,
    )


def merge_bundles(bundles: Sequence[FeatureBundle]) -> FeatureBundle:
    """Concatenate single-graph bundles into one ragged batch.

    The offsets are the same ones ``collate_batches`` applies: ``atom_to_residue``
    shifts by the running residue count and the batch indices by the running graph
    count. Everything else concatenates, because nothing else points at a node.
    """
    if not bundles:
        raise ValueError("cannot merge an empty list of bundles")
    irreps = {b.physics_latent_irreps for b in bundles}
    if len(irreps) != 1:
        raise ValueError(f"bundles disagree on irreps: {sorted(irreps)}")

    residue_offset = 0
    graph_offset = 0
    parts: dict[str, list[Tensor]] = {name: [] for name in _RESIDUE_FIELDS + _ATOM_FIELDS}
    a2r, r_batch, a_batch = [], [], []
    rotation, origin, frame_valid = [], [], []
    for bundle in bundles:
        for name in _RESIDUE_FIELDS + _ATOM_FIELDS:
            parts[name].append(getattr(bundle, name))
        a2r.append(bundle.atom_to_residue + residue_offset)
        r_batch.append(bundle.residue_batch_index + graph_offset)
        a_batch.append(bundle.atom_batch_index + graph_offset)
        rotation.append(bundle.frames.rotation)
        origin.append(bundle.frames.origin)
        frame_valid.append(bundle.frames.valid)
        residue_offset += bundle.num_residues
        graph_offset += bundle.num_graphs

    return replace(
        bundles[0],
        **{name: torch.cat(values) for name, values in parts.items()},
        atom_to_residue=torch.cat(a2r),
        residue_batch_index=torch.cat(r_batch),
        atom_batch_index=torch.cat(a_batch),
        frames=ResidueFrames(
            rotation=torch.cat(rotation),
            origin=torch.cat(origin),
            valid=torch.cat(frame_valid),
        ),
        num_graphs=graph_offset,
    )


# --------------------------------------------------------------------------
# optional on-disk cache
# --------------------------------------------------------------------------


class Phase1FeatureCache:
    """Per-frame frozen features on disk, sharded by domain.

    Optional. The arithmetic decides whether it is worth it: a 118-residue domain
    with ~600 represented atoms costs ~86 kB per frame (152 floats per residue
    plus two 3-vectors per atom), so a 360k-pair manifest would need ~31 GB. For a
    smoke run it is free; for the full run it is a deliberate trade against
    recomputing ~14 ms of GPU per protein per arm.

    **Nothing stale is ever reused.** The shard header records the checkpoint's
    SHA-256, the Phase 1 config hash and the format version; any mismatch raises
    instead of returning features from different weights, which would be
    undetectable downstream.

    Writes are atomic (temp file then rename) and are meant for a single-process
    precompute step, not for DataLoader workers racing on one shard.
    """

    def __init__(self, root: str, *, checkpoint_sha256: str, config_hash: str):
        self.root = root
        self.checkpoint_sha256 = checkpoint_sha256
        self.config_hash = config_hash

    @staticmethod
    def frame_key(domain: str, temperature: str, replica: str, frame: int) -> str:
        """Identity of one cached frame. Frames, not pairs: the same frame is the
        current state of many pairs and is cached once."""
        return f"{domain}/{temperature}/{replica}/{int(frame)}"

    def shard_path(self, domain: str) -> str:
        return os.path.join(self.root, f"{domain}.pt")

    def _header(self) -> dict:
        return {
            "format_version": FEATURE_FORMAT_VERSION,
            "checkpoint_sha256": self.checkpoint_sha256,
            "config_hash": self.config_hash,
        }

    def exists(self, domain: str) -> bool:
        return os.path.exists(self.shard_path(domain))

    def save(self, domain: str, entries: dict[str, FeatureBundle]) -> str:
        """Write one domain's frames atomically. Existing entries are merged in."""
        os.makedirs(self.root, exist_ok=True)
        payload = {"header": self._header(), "entries": {}}
        if self.exists(domain):
            payload["entries"] = dict(self.load(domain).items())
        for key, bundle in entries.items():
            if bundle.num_graphs != 1:
                raise ValueError(
                    f"cache entry {key!r} holds {bundle.num_graphs} graphs; cache one "
                    "frame at a time (use split_bundle)"
                )
            payload["entries"][key] = bundle
        path = self.shard_path(domain)
        tmp = f"{path}.tmp"
        torch.save(payload, tmp)
        os.replace(tmp, path)
        return path

    def load(self, domain: str) -> dict[str, FeatureBundle]:
        """Read one domain's shard, refusing anything built differently."""
        path = self.shard_path(domain)
        payload = torch.load(path, map_location="cpu", weights_only=False)
        header = payload.get("header", {})
        expected = self._header()
        differences = [
            f"{k}: shard has {header.get(k)!r}, this run has {v!r}"
            for k, v in expected.items()
            if header.get(k) != v
        ]
        if differences:
            raise ValueError(
                f"{path} was written by a different Phase 1:\n  "
                + "\n  ".join(differences)
                + "\nDelete the cache and rebuild it; reusing it would mix features "
                "from two sets of weights in one experiment."
            )
        return payload["entries"]

    def get(self, domain: str, keys: Iterable[str]) -> Optional[list[FeatureBundle]]:
        """All requested frames of one domain, or None if any is missing."""
        if not self.exists(domain):
            return None
        entries = self.load(domain)
        out = []
        for key in keys:
            if key not in entries:
                return None
            out.append(entries[key])
        return out
