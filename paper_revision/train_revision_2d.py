"""Revision-only 2D training kernels. No preparation/reconstruction/evaluation imports.

Only the bounded dry-run launcher is released. A formal campaign is not enabled.
All tensors originate from bytes opened by the Python access boundary, never
numpy.fromfile/memmap or torch's filename-based native checkpoint loader.
"""
import copy
import hashlib
import io
import json
from pathlib import Path

import numpy as np
import torch
from bpmr_net.features import B0_OFFSETS, encode_b0_features
from bpmr_net.models import B0Network
from paper_revision.models_revision import build_revision_bpmr


def digest(data):
    return hashlib.sha256(data).hexdigest()


def load_plane(root, row):
    arrays = []
    for kind in ('labels', 'support'):
        payload = (root / row[kind + '_path']).read_bytes()
        if digest(payload) != row[kind + '_sha256']:
            raise ValueError('changed registered plane: ' + row['plane_id'])
        array = np.frombuffer(payload, dtype=np.uint8).reshape(json.loads(row['shape']))
        if row['model_transform'] == 'transpose':
            array = array.T
        arrays.append(array)
    return arrays


def tiny_events(root, rows, material, axis, role, spacing=250):
    candidates = sorted((r for r in rows if r['material'] == material and
                         r['normal_axis'] == axis and r['role'] == role and
                         str(spacing) in r['regimes_um'].split(';')),
                        key=lambda r: int(r['normal_index']))
    for row in candidates:
        labels, support = load_plane(root, row)
        valid = support[4:-4, 4:-4].astype(bool).copy()
        for dr in range(-4, 5):
            for dc in range(-4, 5):
                valid &= support[4+dr:labels.shape[0]-4+dr, 4+dc:labels.shape[1]-4+dc].astype(bool)
        centers = np.argwhere(valid)[:8] + 4
        if len(centers) != 8:
            continue
        rr, cc = centers.T
        x = torch.from_numpy(encode_b0_features(labels, support, rr, cc))
        y = torch.from_numpy(labels[rr, cc].astype(np.int64))
        identity = dict(plane=row['plane_id'], centers=centers.tolist(), role=role,
                        labels_sha256=row['labels_sha256'], support_sha256=row['support_sha256'])
        return x, y, digest(json.dumps(identity, sort_keys=True).encode())
    raise RuntimeError('no eight fully supported registered centers')


def probability(model, x):
    result = model(x)
    if isinstance(result, tuple):
        return result[1]
    return result / result.sum(1, keepdim=True).clamp_min(1e-12)


def loss_for(variant, p, y):
    if variant in {'B0_SOURCE', 'A0', 'A2'}:
        return torch.nn.functional.mse_loss(p, torch.nn.functional.one_hot(y, 4).float())
    return torch.nn.functional.nll_loss(p.clamp_min(1e-12).log(), y)


def save_reload(path, model, metadata):
    stream = io.BytesIO()
    torch.save(dict(state=model.state_dict(), metadata=metadata), stream)
    payload = stream.getvalue()
    path.write_bytes(payload)
    actual = path.read_bytes()
    assert digest(actual) == digest(payload)
    checkpoint = torch.load(io.BytesIO(actual), weights_only=True, map_location='cpu')
    assert checkpoint['metadata'] == metadata
    other = copy.deepcopy(model)
    other.load_state_dict(checkpoint['state'], strict=True)
    assert all(torch.equal(v, other.state_dict()[k]) for k, v in model.state_dict().items())
    return digest(actual)


def require_design_a(metadata, unit, allow_dry_run=False):
    if (metadata.get('design') != 'A' or metadata.get('identity') != 'REVISION_REPLICATION'
            or metadata.get('unit') != unit or metadata.get('variant') != 'B0_SOURCE'
            or metadata.get('historical_checkpoint') is not False
            or metadata.get('kind') != ('DRY_RUN_ONLY' if allow_dry_run else 'EPOCH50')
            or (not allow_dry_run and metadata.get('epochs_completed') != 50)):
        raise PermissionError('not a compatible new Design A source checkpoint')


def dry_run(root, rows, directory, access):
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    results = []
    for material in ('WF2', 'GI'):
        for axis in ('x', 'y', 'z'):
            access.phase = 'registered_2d_event_input'
            x, y, train_hash = tiny_events(root, rows, material, axis, 'train')
            vx, vy, val_hash = tiny_events(root, rows, material, axis, 'validation')
            for seed in (1701, 1702, 1703):
                for template in range(4):
                    unit = f'{material}/G250/{axis}/seed{seed}/template{template}'
                    model_seed = int(digest(unit.encode())[:8], 16)
                    torch.manual_seed(model_seed)
                    source = B0Network()
                    initial = copy.deepcopy(source.state_dict())
                    torch.manual_seed(model_seed)
                    repeat = B0Network()
                    assert all(torch.equal(v, repeat.state_dict()[k]) for k, v in initial.items())
                    common = None
                    common_hash = None
                    for variant in ('B0_SOURCE', 'A0', 'A1', 'A2', 'A3', 'A4', 'A5'):
                        access.phase = variant + ':forward_backward'
                        torch.manual_seed(model_seed)
                        model = source if variant == 'B0_SOURCE' else copy.deepcopy(common)
                        if variant in {'A2', 'A3', 'A4', 'A5'}:
                            model = build_revision_bpmr(model, variant)
                            assert torch.equal(probability(model, x), probability(common, x)) or torch.allclose(probability(model, x), probability(common, x), atol=1e-7)
                        before = {k: v.clone() for k, v in model.state_dict().items()}
                        optimizer = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=.001,
                                                     betas=(.9, .999), eps=1e-8, weight_decay=0)
                        optimizer.zero_grad()
                        loss = loss_for(variant, probability(model, x), y)
                        assert torch.isfinite(loss)
                        loss.backward()
                        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters() if p.requires_grad)
                        optimizer.step()
                        assert any(not torch.equal(before[k], v) for k, v in model.state_dict().items())
                        if variant in {'A2', 'A3', 'A4', 'A5'}:
                            assert all(torch.equal(v, model.b0.state_dict()[k]) for k, v in common.state_dict().items())
                        with torch.no_grad():
                            assert torch.isfinite(loss_for(variant, probability(model, vx), vy))
                        metadata = dict(identity='REVISION_REPLICATION', design='A', kind='DRY_RUN_ONLY',
                                        unit=unit, variant=variant, epochs_completed=0, optimizer_steps=1,
                                        historical_checkpoint=False, train_events=train_hash, validation_events=val_hash,
                                        source_checkpoint_sha256=common_hash)
                        access.phase = variant + ':checkpoint_save_reload'
                        path = directory / (unit.replace('/', '_') + '_' + variant + '.pt')
                        checkpoint_hash = save_reload(path, model, metadata)
                        if variant == 'B0_SOURCE':
                            require_design_a(metadata, unit, allow_dry_run=True)
                            try:
                                require_design_a(metadata, unit)
                            except PermissionError:
                                pass
                            else:
                                raise AssertionError('dry checkpoint accepted for campaign')
                            common = copy.deepcopy(model)
                            common_hash = checkpoint_hash
                        results.append(dict(unit=unit, variant=variant, train_events=train_hash,
                                            validation_events=val_hash, checkpoint_sha256=checkpoint_hash,
                                            source_checkpoint_sha256=metadata['source_checkpoint_sha256']))
    return results


if __name__ == '__main__':
    raise SystemExit('Use python -m paper_revision.audit_training_2d; formal training remains disabled.')
