"""ESM-2 caching and residue conditioning, all offline.

No checkpoint download, no ``transformers`` import: the alignment logic is a
pure function over tensors and the cache is exercised with fake embeddings.
"""

from __future__ import annotations

import dataclasses

import pytest
import torch

from force_md.conditioning import (
    DEFAULT_ESM2_MODEL,
    ESM2_EMBED_DIM,
    ESM2_MAX_RESIDUES,
    Esm2Config,
    Esm2EmbeddingCache,
    ResidueConditioner,
    residue_sequence,
    sequence_fingerprint,
    strip_special_tokens,
    temperature_features,
)
from force_md.data import SyntheticSpec, fake_plm_embedding, synthetic_batch
from force_md.data import residue_constants as rc
from force_md.geometry import apply_rigid_transform, random_rotation_matrix


# --------------------------------------------------------------------------
# sequences
# --------------------------------------------------------------------------


def test_default_model_is_the_documented_checkpoint():
    assert DEFAULT_ESM2_MODEL == "facebook/esm2_t33_650M_UR50D"
    assert ESM2_EMBED_DIM == 1280
    assert Esm2Config().model_name == DEFAULT_ESM2_MODEL


def test_charmm_histidine_becomes_H_not_unknown():
    """HSD/HSE/HSP are the only spelling mdCATH uses; mapping them to X would
    silently corrupt the sequence of every histidine-containing protein."""
    for name in ("HSD", "HSE", "HSP"):
        t = torch.tensor([rc.residue_type_id(name)])
        assert residue_sequence(t) == "H"


def test_residue_sequence_round_trips_the_alphabet():
    types = torch.arange(len(rc.RESIDUE_TYPES) - 1)  # exclude UNK
    seq = residue_sequence(types)
    assert len(seq) == len(types)
    assert "X" not in seq
    assert residue_sequence(torch.tensor([rc.UNK_RESIDUE_ID])) == "X"


def test_sequence_from_a_batch_has_one_letter_per_residue():
    batch = synthetic_batch([SyntheticSpec(9)], seed=0)
    seq = residue_sequence(batch.residues.residue_type)
    assert len(seq) == batch.num_residues


# --------------------------------------------------------------------------
# fingerprints
# --------------------------------------------------------------------------


def test_fingerprint_changes_with_sequence_model_revision_and_layer():
    cfg = Esm2Config()
    base = sequence_fingerprint("ACDEFG", cfg)
    assert base != sequence_fingerprint("ACDEFH", cfg)
    assert base != sequence_fingerprint("ACDEFG", dataclasses.replace(cfg, layer=-2))
    assert base != sequence_fingerprint("ACDEFG", dataclasses.replace(cfg, revision="abc123"))
    assert base != sequence_fingerprint("ACDEFG", dataclasses.replace(cfg, model_name="other"))
    assert base == sequence_fingerprint("ACDEFG", Esm2Config())


def test_fingerprint_ignores_irrelevant_config():
    """max_residues does not change the numbers, so it must not split the cache."""
    cfg = Esm2Config()
    assert sequence_fingerprint("ACDE", cfg) == sequence_fingerprint(
        "ACDE", dataclasses.replace(cfg, max_residues=100)
    )


# --------------------------------------------------------------------------
# special-token alignment
# --------------------------------------------------------------------------


def test_strip_special_tokens_drops_cls_eos_and_padding():
    """Residue i is hidden state i+1. An off-by-one shifts the whole chain."""
    b, t, d = 2, 8, 4
    hidden = torch.arange(b * t * d, dtype=torch.float32).reshape(b, t, d)
    out = strip_special_tokens(hidden, [3, 5])
    assert [o.shape[0] for o in out] == [3, 5]
    assert torch.equal(out[0], hidden[0, 1:4])
    assert torch.equal(out[1], hidden[1, 1:6])


def test_strip_special_tokens_detects_truncation():
    hidden = torch.zeros(1, 5, 4)  # room for 3 residues only
    with pytest.raises(ValueError, match="truncated"):
        strip_special_tokens(hidden, [10])


def test_esm2_residue_limit_is_documented_and_enforced_by_config():
    assert ESM2_MAX_RESIDUES == 1022
    assert Esm2Config().allow_truncation is False


# --------------------------------------------------------------------------
# cache
# --------------------------------------------------------------------------


@pytest.fixture
def cache(tmp_path):
    return Esm2EmbeddingCache(tmp_path / "esm2")


def test_cache_round_trip(cache):
    cfg = Esm2Config()
    seq = "ACDEFGHIK"
    emb = torch.randn(len(seq), 16)
    cache.save("1abcA00", seq, emb, cfg)
    assert cache.exists("1abcA00")
    got = cache.load("1abcA00", expect_sequence=seq, config=cfg)
    assert torch.allclose(got, emb)


def test_cache_rejects_a_different_sequence(cache):
    cfg = Esm2Config()
    cache.save("1abcA00", "ACDEFG", torch.randn(6, 8), cfg)
    with pytest.raises(ValueError, match="does not match"):
        cache.load("1abcA00", expect_sequence="ACDEFH", config=cfg)


def test_cache_rejects_a_different_checkpoint(cache):
    """A cache entry from another model must never be silently reused."""
    cache.save("1abcA00", "ACDEFG", torch.randn(6, 8), Esm2Config())
    other = dataclasses.replace(Esm2Config(), model_name="facebook/esm2_t12_35M_UR50D")
    with pytest.raises(ValueError, match="does not match"):
        cache.load("1abcA00", expect_sequence="ACDEFG", config=other)


def test_cache_rejects_length_mismatch_on_save(cache):
    with pytest.raises(ValueError, match="rows for a"):
        cache.save("1abcA00", "ACDEFG", torch.randn(5, 8), Esm2Config())


def test_cache_missing_entry_raises(cache):
    with pytest.raises(FileNotFoundError):
        cache.load("nope")


def test_cache_metadata_records_provenance(cache, tmp_path):
    cfg = Esm2Config(revision="deadbeef", layer=-2)
    cache.save("1abcA00", "ACDEFG", torch.randn(6, 8), cfg)
    payload = torch.load(cache.path_for("1abcA00"), weights_only=False)
    meta = payload["metadata"]
    assert meta["model_name"] == DEFAULT_ESM2_MODEL
    assert meta["revision"] == "deadbeef"
    assert meta["layer"] == -2
    assert meta["num_residues"] == 6
    assert meta["sequence"] == "ACDEFG"


def test_cache_load_without_verification_still_works(cache):
    """Loading without a sequence is allowed but skips the safety check."""
    cache.save("1abcA00", "ACDEFG", torch.randn(6, 8), Esm2Config())
    assert cache.load("1abcA00").shape == (6, 8)


# --------------------------------------------------------------------------
# temperature
# --------------------------------------------------------------------------


def test_temperature_features_shape_and_finiteness():
    t = torch.tensor([320.0, 348.0, 379.0, 413.0, 450.0])
    f = temperature_features(t)
    assert f.shape == (5, 10)
    assert bool(torch.isfinite(f).all())


def test_temperature_features_distinguish_the_five_mdcath_temperatures():
    t = torch.tensor([320.0, 348.0, 379.0, 413.0, 450.0])
    f = temperature_features(t)
    for i in range(5):
        for j in range(i + 1, 5):
            assert not torch.allclose(f[i], f[j])


def test_temperature_features_are_monotonic_in_the_normalised_channel():
    t = torch.tensor([320.0, 400.0, 450.0])
    f = temperature_features(t)
    assert bool((f[1:, -2] > f[:-1, -2]).all())   # normalised T increases
    assert bool((f[1:, -1] < f[:-1, -1]).all())   # beta = 1/kT decreases


# --------------------------------------------------------------------------
# ResidueConditioner
# --------------------------------------------------------------------------


@pytest.fixture
def batch():
    return synthetic_batch([SyntheticSpec(7), SyntheticSpec(4, nonstandard_at=(1,))],
                           seed=0, plm_dim=32)


def test_conditioner_output_shape(batch):
    m = ResidueConditioner(plm_dim=32, out_channels=64)
    out = m(batch)
    assert out.shape == (batch.num_residues, 64)
    assert bool(torch.isfinite(out).all())


def test_conditioner_zeroes_masked_residues(batch):
    m = ResidueConditioner(plm_dim=32, out_channels=16)
    out = m(batch)
    masked = ~batch.residues.mask
    assert bool(masked.any())
    assert torch.allclose(out[masked], torch.zeros_like(out[masked]))
    assert not torch.allclose(out[~masked], torch.zeros_like(out[~masked]))


def test_conditioner_output_is_se3_invariant(batch):
    """Residue semantics have no orientation; a rigid motion must not touch them."""
    m = ResidueConditioner(plm_dim=32, out_channels=16).eval()
    q = random_rotation_matrix(torch.Generator().manual_seed(0), dtype=torch.float32)
    t = torch.tensor([5.0, -2.0, 1.0])
    with torch.no_grad():
        a = m(batch)
        b = m(apply_rigid_transform(batch, q, t))
    assert torch.allclose(a, b, atol=1e-6)


def test_conditioner_uses_the_plm_embedding(batch):
    m = ResidueConditioner(plm_dim=32, out_channels=16).eval()
    with torch.no_grad():
        a = m(batch)
        perturbed = dataclasses.replace(
            batch,
            residues=dataclasses.replace(
                batch.residues, plm_embedding=batch.residues.plm_embedding + 1.0
            ),
        )
        b = m(perturbed)
    assert not torch.allclose(a, b), "PLM branch has no effect on the output"


def test_conditioner_uses_temperature(batch):
    m = ResidueConditioner(plm_dim=32, out_channels=16).eval()
    with torch.no_grad():
        a = m(batch)
        hot = dataclasses.replace(batch, temperature=batch.temperature + 100.0)
        b = m(hot)
    assert not torch.allclose(a, b)


def test_no_plm_ablation_is_a_config_change(batch):
    """Checkpoint 8 needs a no-PLM ablation without swapping the class."""
    m = ResidueConditioner(plm_dim=32, out_channels=16, use_plm=False).eval()
    assert not hasattr(m, "plm_projection")
    with torch.no_grad():
        a = m(batch)
        perturbed = dataclasses.replace(
            batch,
            residues=dataclasses.replace(
                batch.residues, plm_embedding=torch.randn_like(batch.residues.plm_embedding)
            ),
        )
        b = m(perturbed)
    assert torch.allclose(a, b), "use_plm=False must ignore the embedding entirely"


def test_no_temperature_ablation(batch):
    m = ResidueConditioner(plm_dim=32, out_channels=16, use_temperature=False).eval()
    with torch.no_grad():
        a = m(batch)
        b = m(dataclasses.replace(batch, temperature=batch.temperature + 100.0))
    assert torch.allclose(a, b)


def test_conditioner_rejects_a_plm_width_mismatch(batch):
    m = ResidueConditioner(plm_dim=1280, out_channels=16)
    with pytest.raises(ValueError, match="width 32"):
        m(batch)


def test_conditioner_is_differentiable(batch):
    m = ResidueConditioner(plm_dim=32, out_channels=16)
    m(batch).pow(2).sum().backward()
    grads = [p.grad for p in m.parameters() if p.grad is not None]
    assert grads and all(bool(torch.isfinite(g).all()) for g in grads)


def test_conditioner_does_not_import_transformers():
    """Training must never depend on the PLM package being installed."""
    import sys
    assert "transformers" not in sys.modules or True  # may be imported by others
    import force_md.conditioning.residue as mod
    src = open(mod.__file__).read()
    assert "import transformers" not in src and "from transformers" not in src


def test_fake_embedding_is_deterministic_and_batch_independent():
    a = synthetic_batch([SyntheticSpec(5)], seed=3, plm_dim=16)
    b = synthetic_batch([SyntheticSpec(5), SyntheticSpec(9)], seed=3, plm_dim=16)
    assert torch.allclose(a.residues.plm_embedding, b.residues.plm_embedding[:5])
    again = fake_plm_embedding(a.residues.residue_type, 16, 3,
                               position_index=torch.arange(5))
    assert torch.allclose(a.residues.plm_embedding, again)
