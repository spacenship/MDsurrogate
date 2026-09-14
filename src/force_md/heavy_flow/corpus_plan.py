"""Immutable, revision-pinned corpus order and domain splits for disk rotation."""
from __future__ import annotations

import hashlib
import json
import math
import random
from pathlib import Path


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def domain_name(path: str) -> str:
    if Path(path).is_absolute() or '..' in Path(path).parts:
        raise ValueError(f"unsafe shard path: {path}")
    name = Path(path).name
    if not name.startswith("mdcath_dataset_") or not name.endswith(".h5"):
        raise ValueError(f"unexpected shard path: {path}")
    domain = name[len("mdcath_dataset_"):-3]
    if not domain or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in domain):
        raise ValueError(f"unsafe domain: {domain}")
    return domain


def build_plan(shards, *, repo_id, revision, seed=0, chunk_size=500,
               validation_fraction=0.1, test_fraction=0.1, checkpoint=None):
    if chunk_size < 1 or not all(math.isfinite(x) and 0 <= x < 1 for x in (validation_fraction, test_fraction)) or validation_fraction + test_fraction >= 1:
        raise ValueError("invalid chunk size or split fractions")
    entries = sorted((dict(path=p, size=int(s), domain=domain_name(p)) for p, s in shards), key=lambda x: x['path'])
    if not entries or len({e['domain'] for e in entries}) != len(entries) or any(e['size'] <= 0 for e in entries):
        raise ValueError("empty, duplicate, or invalid shard catalog")
    random.Random(seed).shuffle(entries)
    locked_train, locked_validation, heldout_frames = set(), set(), []
    legacy = None
    if checkpoint is not None:
        rotation = checkpoint['chunk_rotation']
        if rotation.get('step_in_chunk', 0) != 0 or rotation['next_chunk'] != rotation['total_chunks']:
            raise ValueError("import requires a completed legacy rotation checkpoint")
        if rotation['chunk_size'] != chunk_size or rotation.get('corpus_plan_digest'):
            raise ValueError("import requires matching chunk size and a legacy checkpoint")
        split = checkpoint['split_manifest']
        # Include all historical train domains, including normalizer-fit data.
        locked_train = {x['domain'] for x in split['train']}
        locked_validation = {x['domain'] for x in split['unseen_domain_validation']}
        same_validation = split['same_domain_validation']
        locked_validation |= {x['domain'] for x in same_validation} - locked_train
        if locked_train & locked_validation:
            raise ValueError("legacy train/unseen validation domains overlap")
        heldout_frames = same_validation
        prefix = rotation['domain_order']
        if len(prefix) % chunk_size:
            raise ValueError("legacy prefix must end at a full chunk boundary")
        lookup = {e['domain']: e for e in entries}
        if len(set(prefix)) != len(prefix) or not set(prefix) <= lookup.keys():
            raise ValueError("legacy prefix does not match remote catalog")
        if not (locked_train | locked_validation) <= set(prefix):
            raise ValueError("legacy split contains domains outside its catalog")
        entries = [lookup[d] for d in prefix] + [e for e in entries if e['domain'] not in set(prefix)]
        legacy = {'domain_order': prefix, 'step': checkpoint['step'],
                  'normalizer': checkpoint['normalizer'], 'split_digest': digest(split),
                  'next_chunk': len(prefix) // chunk_size}
    available = [e['domain'] for e in entries if e['domain'] not in locked_train | locked_validation]
    random.Random(seed + 1).shuffle(available)
    n = len(entries)
    nt = round(n * test_fraction)
    nv = max(0, round(n * validation_fraction) - len(locked_validation))
    if nt + nv > len(available):
        raise ValueError("historical training leaves too few domains for requested heldout fractions")
    test = set(available[:nt])
    validation = locked_validation | set(available[nt:nt + nv])
    splits = {'train': [], 'validation': [], 'test': []}
    for e in entries:
        role = 'test' if e['domain'] in test else 'validation' if e['domain'] in validation else 'train'
        splits[role].append(e['domain'])
    body = dict(version=1, repo_id=repo_id, revision=revision, seed=seed, chunk_size=chunk_size,
                validation_fraction=validation_fraction, test_fraction=test_fraction,
                shards=entries, splits=splits, legacy=legacy, legacy_validation_frames=heldout_frames)
    return {**body, 'digest': digest(body)}


def validate_plan(plan):
    body = {k: v for k, v in plan.items() if k != 'digest'}
    if plan.get('version') != 1 or digest(body) != plan.get('digest'):
        raise ValueError("corpus plan digest/version mismatch")
    domains = [domain_name(e['path']) for e in plan['shards']]
    if domains != [e['domain'] for e in plan['shards']] or len(set(domains)) != len(domains):
        raise ValueError("invalid corpus domain catalog")
    groups = [set(plan['splits'][k]) for k in ('train', 'validation', 'test')]
    if any(groups[i] & groups[j] for i in range(3) for j in range(i)) or set.union(*groups) != set(domains):
        raise ValueError("splits must be disjoint and cover the catalog")
    if not plan['revision'] or plan['chunk_size'] < 1:
        raise ValueError("missing revision or invalid chunk size")
    return plan


def load_plan(path):
    return validate_plan(json.loads(Path(path).read_text()))


def chunk_entries(plan, index):
    if index < 0:
        raise ValueError("negative chunk index")
    start = index * plan['chunk_size']
    return plan['shards'][start:start + plan['chunk_size']]


def fixed_frame_split(index, plan, seed=0):
    from .physics_dataset import PhysicsFrameKey, PhysicsSplitManifest
    roles = {d: k for k, ds in plan['splits'].items() for d in ds}
    held = {PhysicsFrameKey.from_value(x) for x in plan['legacy_validation_frames']}
    grouped = {'train': [], 'validation': [], 'test': []}
    for value in sorted({PhysicsFrameKey.from_value(x) for x in index}):
        role = roles[value.domain]
        grouped['validation' if value in held else role].append(value)
    # Existing Stage 3 schema has two evaluation slots. Labels are explicit.
    return PhysicsSplitManifest(grouped['train'], grouped['validation'], grouped['test'], seed=seed,
        metadata={'split_type': 'fixed_corpus_domains', 'corpus_plan_digest': plan['digest'],
                  'same_domain_validation_role': 'validation', 'unseen_domain_validation_role': 'test'})
