import copy
import inspect
import json
import math
import random
from pathlib import Path
import numpy as np
from PIL import Image
import pytest
import torch
from torch import nn
from torch.nn import functional as F
from models import SDDNetV1, ConditionalAffine
from losses import SDDLoss
from losses.mining import select_triplets
from datasets.sampling import AnchorSampler
from engine.training import window_sample_count
from engine.condition_diagnostics import derangement, run_condition_diagnostics, donor_indices
from utils.experiment import experiment_config, training_contract, architecture_hash, object_hash, validate_checkpoint
from utils.io import read_csv
from utils.ablation_metrics import paired_summary

@pytest.fixture
def mill_root(tmp_path):
    pixels = np.arange(32 * 40 * 3, dtype=np.uint16).reshape(32, 40, 3)
    for (s, offset) in (('train', 0), ('validation', 101)):
        for folder in ('input', 'gt'):
            (tmp_path / s / folder).mkdir(parents=True)
        for i in range(4):
            content = f'{s}_content{i}'
            image = Image.fromarray(((pixels + offset + i * 3) % 256).astype(np.uint8))
            image.save(tmp_path / s / 'gt' / f'{content}.png')
            for level in (1, 5, 10):
                image.save(tmp_path / s / 'input' / f'{content}-{level}.png')
    return tmp_path

@pytest.mark.updates
def test_variable_tail_accumulation_gradient_equivalence():
    torch.manual_seed(3)
    original = nn.Linear(3, 2).double()
    accum = copy.deepcopy(original)
    full = copy.deepcopy(original)
    aopt = torch.optim.SGD(accum.parameters(), lr=0.01)
    fopt = torch.optim.SGD(full.parameters(), lr=0.01)
    counts = [8, 2, 8]
    xs = [torch.randn(n, 3, dtype=torch.float64) for n in counts]
    ys = [torch.randn(n, 2, dtype=torch.float64) for n in counts]
    for start in (0, 2):
        aopt.zero_grad()
        fopt.zero_grad()
        for i in range(start, min(start + 2, len(counts))):
            F.mse_loss(accum(xs[i]), ys[i]).mul(counts[i] / window_sample_count(counts, 2, i)).backward()
        F.mse_loss(full(torch.cat(xs[start:start + 2])), torch.cat(ys[start:start + 2])).backward()
        for (ap, fp) in zip(accum.parameters(), full.parameters()):
            torch.testing.assert_close(ap.grad, fp.grad, rtol=1e-14, atol=1e-14)
        aopt.step()
        fopt.step()
    for (a, b) in zip(accum.parameters(), full.parameters()):
        torch.testing.assert_close(a, b, rtol=1e-14, atol=1e-14)

def test_fixed_validation_donors_no_fixed_points_and_state_restoration(mill_root, tmp_path):
    from datasets.paired_dataset import MILLPairedDataset
    ds = MILLPairedDataset(mill_root, 'validation')
    from datasets.common import mill_records
    ds.rows = [dict(filename=r.filename, content_id=r.content_id, level=r.level, split='validation', group_id='') for r in mill_records(mill_root, 'validation')]
    model = SDDNetV1().train()
    before = {k: v.detach().clone() for (k, v) in model.state_dict().items()}
    rng = torch.get_rng_state().clone()
    result = run_condition_diagnostics(model, ds, tmp_path / 'diagnostic', size=16, contents=2)
    assert model.training and result['restored_exactly']
    torch.testing.assert_close(rng, torch.get_rng_state(), rtol=0, atol=0)
    for (k, v) in before.items():
        torch.testing.assert_close(v, model.state_dict()[k], rtol=0, atol=0)
    rows = read_csv(tmp_path / 'diagnostic/condition_per_image.csv')
    for row in rows:
        if row['donor_filename']:
            assert row['filename'] != row['donor_filename']
        assert row['split'] == 'validation'
    normal = [r for r in rows if r['mode'] == 'normal']
    assert len(normal) == 6 and all((float(r['raw_output_MAE']) == 0 for r in normal))
    for count in (2, 3, 30):
        assert all((i != j for (i, j) in enumerate(derangement(count))))
    with pytest.raises(ValueError):
        derangement(1)

def test_development_data_audit_skips_test_lol_and_detects_content_leak(mill_root, tmp_path):
    from scripts.prepare_data import prepare_data
    cfg = experiment_config('configs/a0.yaml')
    cfg['run_dir'] = str(tmp_path / 'audit')
    cfg['data'].update(mill_root=str(mill_root), lolv1_root='does_not_exist', lolv2_root='does_not_exist')
    audit = prepare_data(cfg)
    assert audit['passed'] and set(audit['manifest_hashes']) == {'mill'}
    assert set(audit['splits']) == {'mill/train', 'mill/validation'}
    assert all((not overlaps for overlaps in audit['mill_scene_overlaps'].values()))
    image = Image.open(mill_root / 'train/gt/train_content0.png')
    image.save(mill_root / 'validation/gt/validation_content0.png')
    with pytest.raises(RuntimeError, match='audit failed'):
        prepare_data(cfg)

def test_scene_cluster_bootstrap_counts_clusters_not_levels():
    rows = [dict(content_id='a', delta_psnr_vs_input=1), dict(content_id='a', delta_psnr_vs_input=3), dict(content_id='b', delta_psnr_vs_input=-1)]
    result = paired_summary(rows, repeats=100)
    assert result['bootstrap_clusters'] == 2 and result['delta_psnr_vs_input'] == 1
    assert result == paired_summary(rows, repeats=100)
    assert paired_summary([dict(content_id='a', delta_psnr_vs_input=float('inf'))])['delta_psnr_vs_input'] == 'NA'

def test_a0_metadata_statistics_do_not_change_original_loss_or_gradients():
    torch.manual_seed(42)
    original = dict(enhanced=torch.rand(3, 3, 16, 16, requires_grad=True), Fs=torch.rand(3, 128, 2, 2, requires_grad=True), c_hat=torch.rand(3, 1, requires_grad=True))
    measured = {k: v.detach().clone().requires_grad_() for (k, v) in original.items()}
    measured['_batch_info'] = dict(layout='triplet', content_ids=['a', 'a', 'b'], raw_levels=[1, 10, 1], group_ids=[''] * 3)
    gt = torch.rand(3, 3, 16, 16)
    levels = torch.rand(3, 1)
    plain = SDDLoss()(original, gt, levels, progress=2.5)
    observed = SDDLoss()(measured, gt, levels, progress=2.5)
    for key in plain:
        torch.testing.assert_close(plain[key], observed[key], rtol=0, atol=0)
    before = torch.autograd.grad(plain['total'], list(original.values()))
    after = torch.autograd.grad(observed['total'], [measured[k] for k in original])
    for (a, b) in zip(before, after):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
