from types import SimpleNamespace

import torch
from torch import nn

import force_md.heavy_flow.esmc_encoder as esmc_encoder


class _FakeESMCModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(()))
        self.config = SimpleNamespace(hidden_size=5, _commit_hash="resolved-commit")

    def forward(self, input_ids, attention_mask=None):
        del attention_mask
        values = input_ids.to(torch.float32).unsqueeze(-1)
        return SimpleNamespace(last_hidden_state=values.expand(-1, -1, 5) * self.weight)


class _FakeESMCModelClass:
    calls = []

    @classmethod
    def from_pretrained(cls, model_name, **kwargs):
        cls.calls.append((model_name, kwargs))
        return _FakeESMCModel()


class _FakeTokenizer:
    all_special_ids = (101, 102)

    def __call__(self, sequence, *, return_tensors, padding):
        assert return_tensors == "pt"
        assert padding is True
        residue_ids = torch.arange(len(sequence), dtype=torch.int64) + 10
        ids = torch.cat((torch.tensor([101]), residue_ids, torch.tensor([102])))
        return {"input_ids": ids.unsqueeze(0), "attention_mask": torch.ones(1, ids.numel(), dtype=torch.int64)}


def test_hf_esmc_backend_uses_documented_model_and_internal_residue_sequence(monkeypatch):
    fake_module = SimpleNamespace(
        EsmcForMaskedLM=_FakeESMCModelClass,
        EsmcTokenizer=_FakeTokenizer,
    )
    monkeypatch.setattr(esmc_encoder.importlib, "import_module", lambda name: fake_module)
    backend = esmc_encoder._RealESMCBackend(
        esmc_encoder.ESMCConfig(
            model_name="biohub/ESMC-300M",
            revision="main",
            device="cpu",
        )
    )

    # Canonical IDs 0, 1, 19, 20 map to A, R, V, X, not to ESM tokenizer IDs.
    embedding = backend.embed(torch.tensor([0, 1, 19, 20], dtype=torch.int64))
    assert embedding.shape == (4, 5)
    assert backend.model_name == "biohub/ESMC-300M"
    assert backend.revision == "resolved-commit"
    assert backend.embedding_dim == 5
    assert all(not parameter.requires_grad for parameter in backend.parameters())
    assert _FakeESMCModelClass.calls[0][0] == "biohub/ESMC-300M"
    assert _FakeESMCModelClass.calls[0][1]["revision"] == "main"
