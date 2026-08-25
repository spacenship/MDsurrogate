#!/usr/bin/env python
"""Print the whole Phase 1 pipeline on one small protein.

    python examples/hierarchy_sanity.py                # synthetic peptide
    python examples/hierarchy_sanity.py --mdcath       # a real mdCATH frame

Reports hierarchy counts, the tensor shape of everything that crosses a module
boundary, measured equivariance errors, projected force/torque, and the model
outputs. Run it after any change to the geometry, graph or encoder: the
equivariance numbers should stay at ~1e-9 in float64, and a regression shows up
here before it shows up as a bad training run.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from force_md.data import SyntheticSpec, synthetic_batch  # noqa: E402
from force_md.geometry import (  # noqa: E402
    apply_rigid_transform,
    atom_local_coordinates,
    frames_from_batch,
    link_backbone_to_atom_positions,
    random_rotation_matrix,
)
from force_md.graph import (  # noqa: E402
    build_hierarchical_graph,
    edge_geometry,
    edge_spherical_harmonics,
    merge_edge_sets,
)
from force_md.models import LocalPhysicsConfig, LocalPhysicsModel  # noqa: E402
from force_md.nn import EncoderConfig, extract_scalars  # noqa: E402
from force_md.physics import (  # noqa: E402
    ResidueSumProjector,
    omitted_atom_residual,
)

RULE = "=" * 78


def section(title: str) -> None:
    print(f"\n{RULE}\n{title}\n{RULE}")


def load_batch(use_mdcath: bool, root: Path):
    if not use_mdcath:
        return synthetic_batch(
            [SyntheticSpec(12)], seed=0, include_hydrogens=True,
            plm_dim=32, dtype=torch.float64,
        ), 32
    from force_md.data.adapters import MdCathConfig, MdCathDataset

    cache = root / "esm2_cache"
    config = MdCathConfig(
        data_dir=str(root / "data"),
        esm2_cache_dir=str(cache) if cache.is_dir() else None,
        quarantine_path=str(root / "mdcath_force_quarantine.json"),
        max_domains=1, max_residues=120, frames_per_trajectory=1,
        temperatures=(320,), replicas=(4,),
        allow_fake_plm=not cache.is_dir(),
        represented_scope="all_atom",  # keep H so the residual is visible
        dtype=torch.float64,
    )
    dataset = MdCathDataset(config)
    if len(dataset) == 0:
        raise SystemExit("no mdCATH frame matched the filters")
    return dataset[0].batch, dataset[0].batch.residues.plm_dim


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser()
    ap.add_argument("--mdcath", action="store_true", help="use a real mdCATH frame")
    args = ap.parse_args()

    torch.manual_seed(0)
    batch, plm_dim = load_batch(args.mdcath, root)
    batch = link_backbone_to_atom_positions(batch)
    batch.validate()

    # ---------------------------------------------------------------- counts
    section("1. HIERARCHY")
    print(f"  domain(s)              {batch.domain_id}")
    print(f"  temperature            {batch.temperature.tolist()} K")
    print(f"  backbone frame nodes   {batch.num_residues}")
    print(f"  residue semantic nodes {batch.num_residues}")
    print(f"  atom child nodes       {batch.num_atoms}"
          f"  (heavy {int(batch.atoms.is_heavy.sum())}, "
          f"H {int((~batch.atoms.is_heavy).sum())})")
    counts = torch.bincount(batch.atoms.atom_to_residue)
    print(f"  atoms per residue      min {int(counts.min())} max {int(counts.max())}"
          f"  <- ragged, not padded")
    print(f"  units                  {batch.units.length} / {batch.units.force}")
    print(f"  frame_index            {batch.frame_index.tolist()}  "
          "(frames, NOT nanoseconds: mdCATH stores no timestamp)")

    # ---------------------------------------------------------------- frames
    section("2. RESIDUE FRAMES AND LOCAL COORDINATES")
    frames = frames_from_batch(batch)
    local, _ = atom_local_coordinates(batch, frames)
    eye = torch.eye(3, dtype=frames.rotation.dtype).expand_as(frames.rotation)
    det = torch.linalg.det(frames.rotation)
    print(f"  rotation R_i           {tuple(frames.rotation.shape)}  columns = e1|e2|e3")
    print(f"  origin r_i (= CA)      {tuple(frames.origin.shape)}")
    print(f"  valid frames           {int(frames.valid.sum())}/{frames.num_residues}")
    print(f"  max |R^T R - I|        {(frames.rotation.transpose(-1,-2) @ frames.rotation - eye).abs().max():.3e}")
    print(f"  det(R) range           [{det.min():.12f}, {det.max():.12f}]")
    print(f"  y_ia local coords      {tuple(local.shape)}  "
          f"|y| mean {local.norm(dim=-1).mean():.2f} A")

    # ----------------------------------------------------------------- graph
    section("3. TOPOLOGY (6 relations)")
    graph = build_hierarchical_graph(batch)
    graph.validate(batch)
    for name, edges in graph.relations().items():
        types = torch.bincount(edges.edge_type, minlength=edges.num_types).tolist()
        print(f"  {name:32s} {edges.num_edges:7d} edges   sub-types {types}")
    atom_edges = merge_edge_sets([graph.atom_bonded, graph.atom_spatial], "m")
    geom = edge_geometry(batch.atoms.positions, batch.atoms.positions, atom_edges)
    sh = edge_spherical_harmonics(geom.unit_vector, 2)
    print(f"\n  edge distance                {tuple(geom.distance.shape)}   "
          f"invariant, {geom.distance.min():.2f}-{geom.distance.max():.2f} A")
    print(f"  edge vector                  {tuple(geom.vector.shape)}   equivariant l=1")
    print(f"  edge spherical harmonics     {tuple(sh.shape)}   1x0e+1x1o+1x2e")

    # ------------------------------------------------------------ projection
    section("4. FORCE PROJECTION  (F_i = sum f_a,  tau_i = sum (x_a - r_i) x f_a)")
    heavy = ResidueSumProjector("heavy_atom")(batch)
    all_atom = ResidueSumProjector("all_atom")(batch)
    print(f"  heavy-atom  |F| mean   {heavy.force.norm(dim=-1).mean():9.3f}   "
          f"|tau| mean {heavy.torque.norm(dim=-1).mean():9.3f}")
    print(f"  all-atom    |F| mean   {all_atom.force.norm(dim=-1).mean():9.3f}   "
          f"|tau| mean {all_atom.torque.norm(dim=-1).mean():9.3f}")
    if int((~batch.atoms.is_heavy).sum()) > 0:
        residual, _ = omitted_atom_residual(all_atom, heavy)
        ratio = float(residual.norm(dim=-1).mean() / heavy.force.norm(dim=-1).mean())
        print(f"  omitted-atom residual  {residual.norm(dim=-1).mean():9.3f}   "
              f"= {ratio:.2f} x the heavy-only force")
        print("    identifiable only because mdCATH stores hydrogens; solvent is")
        print("    NOT part of this residual and stays in the uncertainty.")
    print(f"  torque origin          CA  (a torque without its origin is ambiguous)")

    # ----------------------------------------------------------------- model
    section("5. MODEL OUTPUT")
    model = LocalPhysicsModel(
        LocalPhysicsConfig(encoder=EncoderConfig(plm_dim=plm_dim))
    ).to(torch.float64).eval()
    out = model(batch)
    print(f"  parameters             {sum(p.numel() for p in model.parameters()):,}")
    for name in ("atom_force_mean", "atom_force_residual", "atom_force_conservative",
                 "atom_force_logvar", "residue_explained_force", "residue_hidden_force",
                 "residue_force_mean", "residue_torque_mean", "residue_force_logvar",
                 "aggregated_atom_force", "energy", "residue_energy", "physics_latent"):
        value = getattr(out, name)
        shape = "None" if value is None else str(tuple(value.shape))
        print(f"  {name:26s} {shape}")
    print(f"  physics_latent irreps  {out.physics_latent_irreps}")
    print(f"  target_scope           {out.target_scope}")

    # ---------------------------------------------------------- equivariance
    section("6. SYMMETRY (float64)")
    q = random_rotation_matrix(torch.Generator().manual_seed(0), dtype=torch.float64)
    t = torch.tensor([3.0, -7.5, 0.25], dtype=torch.float64)
    moved = link_backbone_to_atom_positions(apply_rigid_transform(batch, q, t))
    out2 = model(moved)

    d = model.irreps.D_from_matrix(q)
    latent_err = (out2.physics_latent - out.physics_latent @ d.T).abs().max()
    scale = out.physics_latent.abs().max()
    print(f"  physics_latent    max|Y' - Y D^T| / |Y|   {float(latent_err/scale):.3e}")
    for name in ("atom_force_mean", "residue_force_mean", "residue_torque_mean"):
        x, y = getattr(out, name), getattr(out2, name)
        rel = float((y - x @ q.T).abs().max() / max(float(x.abs().max()), 1e-30))
        print(f"  {name:26s} equivariance rel err  {rel:.3e}")
    s0 = extract_scalars(out.physics_latent, model.irreps)
    s1 = extract_scalars(out2.physics_latent, model.irreps)
    print(f"  l=0 block         max|s' - s|             {float((s1-s0).abs().max()):.3e}")
    print(f"  energy            max|U' - U|             {float((out2.energy-out.energy).abs().max()):.3e}")
    print(f"  logvar            max|v' - v|             "
          f"{float((out2.atom_force_logvar-out.atom_force_logvar).abs().max()):.3e}")

    mirror = torch.diag(torch.tensor([-1.0, 1.0, 1.0], dtype=torch.float64))
    mirrored = link_backbone_to_atom_positions(
        apply_rigid_transform(batch, mirror, torch.zeros(3, dtype=torch.float64))
    )
    s_mirror = extract_scalars(model(mirrored).physics_latent, model.irreps)
    print(f"\n  reflection: l=0 block changes by {float((s_mirror-s0).abs().max()):.3f}")
    print("    Reflection is NOT a symmetry here. Chirality enters through the")
    print("    residue-frame local coordinates, so an L-protein and its D mirror")
    print("    image produce different features -- as they must.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
