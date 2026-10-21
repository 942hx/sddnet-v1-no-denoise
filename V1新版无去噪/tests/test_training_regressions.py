"""Small CPU/mocked integration regressions; all outputs live in pytest temp dirs.

The only CUDA cases use a disposable tiny model, not the real 100-epoch run.
No real datasets, saved experiment checkpoints, or historical events are written.
"""
from __future__ import annotations
import copy
import math
from types import SimpleNamespace
from unittest.mock import Mock
import pytest
import torch
import yaml
from torch import nn
from torch.utils.data import Dataset
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from engine import training, evaluation, reporting
from engine.diagnostics import branch_gradient_norms
from scripts import train as run_pipeline
from utils.experiment import experiment_config, training_contract, validate_checkpoint, validate_formal_checkpoint, read_json, architecture_hash, object_hash
from utils.io import read_csv, write_csv
from utils.runtime import load_checkpoint, save_checkpoint, make_scaler
BRANCHES = ('structure_branch', 'degradation_branch', 'illumination_head', 'bottleneck_affine', 'skip_affine2', 'skip_affine1', 'skip_affine0', 'head')
TRAINER_CLASS = training.Trainer

class TinyModel(nn.Module):
    """Every named branch participates, with cheap deterministic CPU gradients."""

    def __init__(self, **options):
        super().__init__()
        if options:
            assert options == dict(affine_mode='bounded',local_init_std=.001,residual_gate=False,g_min=.25)
        for name in BRANCHES:
            setattr(self, name, nn.Linear(3, 3))

    def forward(self, x, return_latents=False):
        feature = x.mean((2, 3))
        rgb = torch.stack([getattr(self, name)(feature) for name in BRANCHES]).mean(0).sigmoid()
        return {'enhanced': rgb[:, :, None, None].expand_as(x)}

class TinyLoss(nn.Module):

    def forward(self, output, gt, levels, progress):
        loss = (output['enhanced'].float() - gt.float()).square().mean()
        return dict(total=loss, rec=loss)

class Triplets(Dataset):

    def __init__(self, count):
        self.count = count

    def __len__(self):
        return self.count

    def __getitem__(self, index):
        result = {}
        for (role, offset) in (('anchor', 0), ('positive', 1), ('negative', 2)):
            result[role + '_low'] = torch.full((3, 12, 12), 0.1 + 0.01 * ((index + offset) % 10))
            result[role + '_gt'] = torch.full((3, 12, 12), 0.6 + 0.01 * offset)
            result[role + '_level'] = torch.tensor([0.2], dtype=torch.float32)
        return result

def config(tmp_path, **changes):
    cfg = experiment_config('configs/a0.yaml')
    cfg.update(run_dir=str(tmp_path), device='cpu', amp=False, cpu_threads=2)
    cfg.update(changes)
    return cfg

def trainer_factory(monkeypatch, tmp_path, count=12, interval=100, device='cpu'):
    monkeypatch.setattr(training, 'manifest_identity', lambda cfg: {'fixture': 'unit'})
    monkeypatch.setattr(training, 'MILLTripletDataset', lambda *a, **k: Triplets(count))
    monkeypatch.setattr(training, 'ManifestPairedDataset', lambda *a, **k: [])
    monkeypatch.setattr(training, 'SDDNetV1', TinyModel)
    monkeypatch.setattr(training, 'SDDLoss', lambda **k: TinyLoss())
    cfg = config(tmp_path, device=device, amp=device == 'cuda')
    cfg['diagnostics']['grad_norm_every'] = interval
    return TRAINER_CLASS(cfg)

class SkippingScaler:
    """CPU simulation of scale/unscale, found_inf and skipped optimizer updates."""

    def __init__(self, skip=(2,)):
        self._scale = torch.tensor(8.0)
        self.flag = torch.tensor(0.0)
        self.skip = set(skip)
        self.attempt = 0

    def is_enabled(self):
        return True

    def get_scale(self):
        return self._scale.item()

    def _get_scale_async(self):
        return self._scale

    def _found_inf_per_device(self, optimizer):
        return {'cpu': self.flag}

    def scale(self, loss):
        return loss * self._scale

    def unscale_(self, optimizer):
        self.attempt += 1
        self.flag.fill_(float(self.attempt in self.skip))
        for group in optimizer.param_groups:
            for parameter in group['params']:
                if parameter.grad is not None:
                    parameter.grad.div_(self._scale)

    def step(self, optimizer):
        if self.attempt not in self.skip:
            optimizer.step()

    def update(self):
        if self.attempt in self.skip:
            self._scale.div_(2)

@pytest.mark.updates
@pytest.mark.parametrize('interval,count,expected', [(0, 12, []), (1, 11, [1, 2, 3, 4, 5, 6]), (100, 400, [100, 200])])
def test_diagnostic_frequency_accumulation_and_missing_values(monkeypatch, tmp_path, interval, count, expected):
    trainer = trainer_factory(monkeypatch, tmp_path, count, interval)
    collect = Mock(wraps=branch_gradient_norms)
    monkeypatch.setattr(training, 'branch_gradient_norms', collect)
    trainer.epoch(1)
    assert collect.call_count == len(expected)
    assert [r['optimizer_step'] for r in trainer.gradient_samples] == expected
    assert trainer.optimizer_updates == math.ceil(count / 2)
    assert trainer.gradient_epoch_metrics()['grad_norm_samples'] == len(expected)
    for row in trainer.logs:
        if not row['branch_grad_norm_sampled']:
            assert 'head_grad_norm_total' not in row
    if not expected:
        assert trainer.last_gradient_norms == {} and trainer.last_gradient_step is None
        assert trainer.gradient_epoch_metrics() == {'grad_norm_samples': 0}
    assert all((p.grad is None for p in trainer.model.parameters()))

@pytest.mark.updates
def test_unsampled_steps_never_enter_amp_probe_or_branch_traversal(monkeypatch, tmp_path):
    trainer = trainer_factory(monkeypatch, tmp_path, count=12, interval=100)
    probe = Mock(side_effect=AssertionError('Non-sampling step entered diagnostic code'))
    monkeypatch.setattr(training, 'sampled_amp_update_state', probe)
    monkeypatch.setattr(training, 'branch_gradient_norms', probe)
    trainer.epoch(1)
    probe.assert_not_called()

@pytest.mark.updates
def test_amp_skip_uses_successful_steps_and_unscaled_preclip_norms(monkeypatch, tmp_path):
    trainer = trainer_factory(monkeypatch, tmp_path, count=6, interval=2)
    trainer.scaler = SkippingScaler()
    trainer.cfg['grad_clip_norm'] = 0.001
    measured = []
    original_clip = torch.nn.utils.clip_grad_norm_

    def collect(model):
        assert trainer.scaler.attempt == 3
        measured.append(branch_gradient_norms(model))
        return measured[-1]

    def clip(parameters, *args, **kwargs):
        if trainer.scaler.attempt == 3:
            assert measured[-1] == branch_gradient_norms(trainer.model)
        return original_clip(parameters, *args, **kwargs)
    monkeypatch.setattr(training, 'branch_gradient_norms', collect)
    monkeypatch.setattr(torch.nn.utils, 'clip_grad_norm_', clip)
    trainer.epoch(1)
    assert trainer.attempted_updates == 3 and trainer.overflows == 1 and (trainer.optimizer_updates == 2)
    assert len(measured) == 1 and trainer.gradient_samples[0]['optimizer_step'] == 2
    assert measured[0]['head_grad_norm_total'] > 0.001
    from torch.utils.tensorboard import SummaryWriter
    with SummaryWriter(str(tmp_path / 'tb')) as writer:
        training.log_gradient_samples(writer, trainer.gradient_samples)
    events = EventAccumulator(str(tmp_path / 'tb')).Reload().Scalars('gradient/head_grad_norm_total')
    assert [event.step for event in events] == [2]
    assert events[0].value == pytest.approx(measured[0]['head_grad_norm_total'])

@pytest.mark.updates
def test_diagnostic_frequency_does_not_change_updates_loss_or_scheduler(monkeypatch, tmp_path):
    results = []
    for interval in (0, 1, 100):
        trainer = trainer_factory(monkeypatch, tmp_path / str(interval), count=8, interval=interval)
        trainer.scaler = SkippingScaler()
        (losses, _) = trainer.epoch(1)
        trainer.scheduler.step()
        results.append((copy.deepcopy(trainer.model.state_dict()), losses, trainer.optimizer.state_dict(), trainer.scheduler.state_dict()))
    for result in results[1:]:
        for (key, value) in results[0][0].items():
            torch.testing.assert_close(value, result[0][key], rtol=0, atol=0)
        assert result[1] == results[0][1] and result[3] == results[0][3]
        for (key, state) in results[0][2]['state'].items():
            for (name, value) in state.items():
                torch.testing.assert_close(value, result[2]['state'][key][name], rtol=0, atol=0)

@pytest.mark.updates
def test_sampling_resume_keeps_global_successful_update_phase(monkeypatch, tmp_path):
    trainer = trainer_factory(monkeypatch, tmp_path, count=4, interval=100)
    trainer.optimizer_updates = 98
    trainer.attempted_updates = 98
    trainer.epoch(1)
    trainer.completed_epoch = 1
    state = trainer.checkpoint()
    state['run_kind'] = 'formal'
    path = tmp_path / 'resume.pth'
    save_checkpoint(path, state)
    restored = trainer_factory(monkeypatch, tmp_path, count=4, interval=1)
    restored.restore(path)
    restored.epoch(2)
    assert [row['optimizer_step'] for row in restored.gradient_samples] == [101, 102]
    assert restored.optimizer_updates == 102
    state.pop('optimizer_updates')
    save_checkpoint(tmp_path / 'missing_counter.pth', state)
    with pytest.raises(ValueError, match='reliable.*optimizer_updates'):
        restored.restore(tmp_path / 'missing_counter.pth')

def test_branch_norms_batch_one_cpu_transfer_without_item(monkeypatch):
    model = TinyModel()
    model(torch.ones(3, 3, 12, 12))['enhanced'].sum().backward()
    original_cpu = torch.Tensor.cpu
    calls = []

    def transfer(self, *args, **kwargs):
        calls.append(tuple(self.shape))
        return original_cpu(self, *args, **kwargs)
    with monkeypatch.context() as patch:
        patch.setattr(torch.Tensor, 'cpu', transfer)
        patch.setattr(torch.Tensor, 'item', Mock(side_effect=AssertionError('Per-branch item synchronization')))
        values = branch_gradient_norms(model)
    assert calls == [(8,)] and all((math.isfinite(value) for value in values.values()))
    model.head.weight.grad = None
    model.head.bias.grad = None
    assert math.isnan(branch_gradient_norms(model)['head_grad_norm_total'])

def test_amp_candidate_fuses_scale_and_skip_flag_in_one_transfer(monkeypatch):
    scaler = SkippingScaler()
    scaler.flag.fill_(1.0)
    original_cpu = torch.Tensor.cpu
    transfers = []

    def transfer(self, *args, **kwargs):
        transfers.append(tuple(self.shape))
        return original_cpu(self, *args, **kwargs)
    with monkeypatch.context() as patch:
        patch.setattr(torch.Tensor, 'cpu', transfer)
        patch.setattr(torch.Tensor, 'item', Mock(side_effect=AssertionError('Extra scalar synchronization')))
        (scale, will_update) = training.sampled_amp_update_state(scaler, None)
    assert scale == 8.0 and (not will_update) and (transfers == [(2,)])

def test_unsupported_scaler_fails_explicitly_instead_of_guessing_skip():
    with pytest.raises(RuntimeError, match='--grad-norm-every 0'):
        training.sampled_amp_update_state(SimpleNamespace(is_enabled=lambda : True), None)

def checkpoint(cfg, kind='formal', epoch=7):
    state = dict(arch_revision='v1_clean', contract=training_contract(cfg), manifest_hashes={'fixture': 'unit'}, model={}, completed_epoch=epoch, sampler={'size': 4, 'position': 4}, variant='A0', seed=42, architecture_hash=architecture_hash(cfg), config_hash=object_hash(training_contract(cfg)))
    if kind is not None:
        state['run_kind'] = kind
    return state

@pytest.mark.parametrize('kind', ['smoke', None, 'unknown'])
def test_direct_evaluator_cannot_bypass_kind_guard(monkeypatch, tmp_path, kind):
    cfg = config(tmp_path / 'run')
    path = tmp_path / 'candidate.pth'
    save_checkpoint(path, checkpoint(cfg, kind))
    forbidden = Mock(side_effect=AssertionError('Evaluator touched data/model before guard'))
    monkeypatch.setattr(evaluation, 'manifest_identity', forbidden)
    monkeypatch.setattr(evaluation, 'SDDNetV1', forbidden)
    with pytest.raises(ValueError, match='formal evaluation'):
        evaluation.evaluate_all(cfg, path)
    forbidden.assert_not_called()
    assert not (tmp_path / 'run').exists()

def test_formal_evaluation_runs_metrics_on_one_temporary_fixture(monkeypatch, tmp_path):
    cfg = config(tmp_path / 'run')
    cfg['evaluation']['save_images'] = False
    path = tmp_path / 'formal.pth'
    save_checkpoint(path, checkpoint(cfg, epoch=0))

    class FixtureModel(nn.Module):

        def __init__(self, **options):
            super().__init__()
            assert options == dict(affine_mode='bounded',local_init_std=.001,residual_gate=False,g_min=.25)

        def forward(self, x, return_latents=False):
            return dict(enhanced=x + 0.1, Fs=torch.ones(1, 1, 1, 1), Fd=torch.ones(1, 1, 1, 1), c_hat=torch.tensor([[0.5]]))

    class FixtureData:
        rows = [dict(filename='fixture.png', scene_id='fixture', illum_level='')]

        def __len__(self):
            return 1

        def __getitem__(self, index):
            return dict(low=torch.full((3, 12, 12), 0.2), gt=torch.full((3, 12, 12), 0.6))
    monkeypatch.setattr(evaluation, 'manifest_identity', lambda cfg: {'fixture': 'unit'})
    monkeypatch.setattr(evaluation, 'DATASETS', ('lolv1',))
    monkeypatch.setattr(evaluation, 'ManifestPairedDataset', lambda *a: FixtureData())
    monkeypatch.setattr(evaluation, 'SDDNetV1', FixtureModel)
    monkeypatch.setattr(evaluation, 'load_lpips', lambda device: (None, dict(available=False, reason='Unit fixture: LPIPS omitted')))
    import scripts.prepare_data
    monkeypatch.setattr(scripts.prepare_data,'prepare_data',lambda *a,**k: None)
    evaluation.evaluate_all(cfg, path)
    rows = read_csv(tmp_path / 'run/metrics/summary.csv')
    assert len(rows) == 2 and rows[-1]['checkpoint_epoch'] == '0'
    assert read_json(tmp_path / 'run/metrics/evaluation_status.json')['complete']

@pytest.mark.parametrize('field,expected', [('arch_revision', 'Legacy resume rejected'), ('contract', 'contract'), ('manifest_hashes', 'fingerprint')])
def test_formal_guard_keeps_existing_metadata_checks(tmp_path, field, expected):
    cfg = config(tmp_path)
    state = checkpoint(cfg)
    state[field] = 'invalid'
    with pytest.raises(ValueError, match=expected):
        validate_formal_checkpoint(state, cfg, {'fixture': 'unit'})

def test_formal_resume_and_generic_smoke_diagnostics_policy(tmp_path):
    cfg = config(tmp_path)
    state = checkpoint(cfg)
    validate_formal_checkpoint(state, cfg, {'fixture': 'unit'}, for_resume=True)
    state['sampler']['position'] = 2
    with pytest.raises(ValueError, match='completed formal epoch'):
        validate_formal_checkpoint(state, cfg, for_resume=True)
    smoke = checkpoint(cfg, 'smoke')
    validate_checkpoint(smoke, cfg, {'fixture': 'unit'})
    with pytest.raises(ValueError, match='smoke checkpoint'):
        validate_formal_checkpoint(smoke, cfg, for_resume=True)

@pytest.mark.parametrize('value', [0, 1, 100])
def test_cli_gradient_option_and_yaml_default(tmp_path, value):
    args = run_pipeline.build_parser().parse_args(['--output', str(tmp_path/'run'), '--grad-norm-every', str(value)])
    assert args.grad_norm_every == value
    assert experiment_config('configs/a0.yaml')['diagnostics']['grad_norm_every'] == 100
    with pytest.raises(SystemExit):
        run_pipeline.build_parser().parse_args(['--output', str(tmp_path/'run'), '--grad-norm-every', '-1'])

@pytest.mark.updates
def test_real_cuda_amp_skip_and_frequency_invariance(monkeypatch, tmp_path):
    if not torch.cuda.is_available():
        pytest.skip('CUDA unavailable; AMP regression not accepted')
    results = []
    for (interval, expected) in ((0, []), (1, [1, 2]), (2, [2])):
        trainer = trainer_factory(monkeypatch, tmp_path / str(interval), count=6, interval=interval, device='cuda')
        trainer.scaler = torch.amp.GradScaler('cuda', init_scale=8.0)
        original = trainer.scaler.unscale_
        attempts = []

        def unscale(optimizer):
            attempts.append(1)
            if len(attempts) == 2:
                next(trainer.model.parameters()).grad.fill_(float('inf'))
            original(optimizer)
        trainer.scaler.unscale_ = unscale
        trainer.epoch(1)
        assert trainer.optimizer_updates == 2 and trainer.overflows == 1
        assert [r['optimizer_step'] for r in trainer.gradient_samples] == expected
        results.append({k: v.detach().cpu().clone() for (k, v) in trainer.model.state_dict().items()})
    for state in results[1:]:
        for (key, value) in results[0].items():
            torch.testing.assert_close(value, state[key], rtol=0, atol=0)

@pytest.mark.updates
def test_real_model_cpu_light_smoke_and_temporary_checkpoint(monkeypatch, tmp_path):
    from models import SDDNetV1
    from losses import SDDLoss
    baseline = None
    for interval in (0, 1):
        trainer = trainer_factory(monkeypatch, tmp_path, count=2, interval=interval)
        trainer.model = SDDNetV1()
        trainer.criterion = SDDLoss()
        trainer.optimizer = training.optimizer_for(trainer.model, trainer.cfg)
        trainer.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(trainer.optimizer, 100, eta_min=1e-06)
        trainer.scaler = make_scaler(False)
        (losses, count) = trainer.epoch(1)
        assert count == 2 and trainer.optimizer_updates == 1 and math.isfinite(losses['total'])
        assert len(trainer.gradient_samples) == interval
        if baseline is None:
            baseline = ({key: value.detach().clone() for (key, value) in trainer.model.state_dict().items()}, losses)
        else:
            assert losses == baseline[1]
            for (key, value) in baseline[0].items():
                torch.testing.assert_close(value, trainer.model.state_dict()[key], rtol=0, atol=0)
    state = trainer.checkpoint()
    state['run_kind'] = 'smoke'
    path = tmp_path / 'smoke.pth'
    save_checkpoint(path, state)
    validate_checkpoint(load_checkpoint(path), trainer.cfg, {'fixture': 'unit'})
    with pytest.raises(ValueError, match='smoke checkpoint'):
        validate_formal_checkpoint(load_checkpoint(path), trainer.cfg)

class CountingCosine(torch.optim.lr_scheduler.CosineAnnealingLR):
    """Count production scheduler calls, excluding its constructor's initial step."""

    def __init__(self, optimizer):
        self.calls = 0
        super().__init__(optimizer, 100, eta_min=1e-06)
        self.calls = 0

    def step(self, *args, **kwargs):
        self.calls += 1
        return super().step(*args, **kwargs)

def run_cpu_formal_driver(monkeypatch, directory, epochs, resume=None):
    """Exercise the real log/checkpoint/TensorBoard driver for just a few epochs.

    Only the CUDA guard, tiny model/data/loss, validation and report are mocked.
    Actual Trainer accumulation/backward/AdamW/scaler/scheduler/CSV/events run.
    The production 100-epoch loop is bounded only in this test's module namespace.
    """
    monkeypatch.setattr(training, 'choose_device', lambda value: torch.device('cpu'))
    trainer = trainer_factory(monkeypatch, directory, count=4, interval=1)
    trainer.optimizer.param_groups[1]['lr'] = 0.0002
    trainer.optimizer.param_groups[1]['initial_lr'] = 0.0002
    trainer.scheduler = CountingCosine(trainer.optimizer)
    used = []
    trainer.optimizer.register_step_pre_hook(lambda opt, args, kwargs: used.append(tuple((g['lr'] for g in opt.param_groups))))

    def construct(cfg, path=None):
        if path:
            trainer.restore(path)
        return trainer
    monkeypatch.setattr(training, 'Trainer', construct)
    monkeypatch.setattr(TRAINER_CLASS, 'diagnose', lambda self: {'scope': 'unit validation fixture'})
    monkeypatch.setattr(training, 'validate', lambda *a, **k: dict(PSNR_RGB=10.0, SSIM_RGB=0.5, count=1, per_image=[], per_level=[]))
    monkeypatch.setattr(reporting, 'report', lambda *a, **k: None)
    monkeypatch.setattr(training, 'range', lambda *args: range(args[0], min(args[1], args[0] + epochs)) if len(args) == 2 and args[1] == 101 else range(*args), raising=False)

    class TorchProxy:
        cuda = SimpleNamespace(is_available=lambda : True)

        def __getattr__(self, name):
            return getattr(torch, name)
    monkeypatch.setattr(training, 'torch', TorchProxy())
    monkeypatch.setattr(training, 'choose_device', lambda value: torch.device('cuda'))
    from engine import condition_diagnostics
    monkeypatch.setattr(condition_diagnostics,'run_condition_diagnostics',lambda *a,**k: None)
    training.formal_training(trainer.cfg, resume)
    return (trainer, used)

@pytest.mark.updates
def test_epoch_lr_csv_tensorboard_groups_and_scheduler_sequence(monkeypatch, tmp_path):
    (trainer, used) = run_cpu_formal_driver(monkeypatch, tmp_path / 'continuous', epochs=4)
    rows = read_csv(tmp_path / 'continuous/history.csv')
    events = EventAccumulator(str(tmp_path / 'continuous/tensorboard')).Reload()
    assert len(rows) == 4 and float(rows[0]['lr']) == 0.0001
    assert trainer.scheduler.calls == 4 and trainer.scheduler.last_epoch == 4
    for (group, initial) in ((0, 0.0001), (1, 0.0002)):
        expected = [1e-06 + (initial - 1e-06) * (1 + math.cos(math.pi * n / 100)) / 2 for n in range(4)]
        key = 'lr' if group == 0 else 'lr_group_1'
        observed = [float(row[key]) for row in rows]
        assert observed == pytest.approx(expected, rel=1e-13)
        assert [value[group] for value in used] == pytest.approx([lr for lr in observed for _ in range(2)])
        tb = events.Scalars('epoch/' + key)
        assert [event.step for event in tb] == [1, 2, 3, 4]
        assert [event.value for event in tb] == pytest.approx(observed, abs=1e-11)
    assert float(rows[-1]['lr']) != trainer.optimizer.param_groups[0]['lr']
    gradients = events.Scalars('gradient/head_grad_norm_total')
    assert [event.step for event in gradients] == list(range(1, 9))
    assert [int(row['grad_norm_samples']) for row in rows] == [2, 2, 2, 2]

@pytest.mark.updates
def test_resumed_lr_and_weights_match_uninterrupted_cpu_driver(monkeypatch, tmp_path):
    (continuous, _) = run_cpu_formal_driver(monkeypatch, tmp_path / 'continuous', epochs=4)
    expected = {key: value.detach().clone() for (key, value) in continuous.model.state_dict().items()}
    (first, _) = run_cpu_formal_driver(monkeypatch, tmp_path / 'resumed', epochs=2)
    next_lr = first.optimizer.param_groups[0]['lr']
    (restored, used) = run_cpu_formal_driver(monkeypatch, tmp_path / 'resumed', epochs=2, resume=tmp_path / 'resumed/checkpoints/last.pth')
    rows = read_csv(tmp_path / 'resumed/history.csv')
    assert [int(row['completed_epoch']) for row in rows] == [1, 2, 3, 4]
    assert float(rows[2]['lr']) == next_lr == used[0][0]
    assert restored.scheduler.calls == 4 and restored.scheduler.last_epoch == 4
    assert [row['lr'] for row in rows] == [row['lr'] for row in read_csv(tmp_path / 'continuous/history.csv')]
    for (key, value) in expected.items():
        torch.testing.assert_close(value, restored.model.state_dict()[key], rtol=0, atol=0)
    tb = EventAccumulator(str(tmp_path / 'resumed/tensorboard')).Reload().Scalars('epoch/lr')
    assert [event.step for event in tb] == [1, 2, 3, 4]
    assert [event.value for event in tb] == pytest.approx([float(row['lr']) for row in rows], abs=1e-11)

def test_direct_formal_training_rejects_smoke_before_constructing_trainer(monkeypatch, tmp_path):
    cfg = config(tmp_path / 'run')
    path = tmp_path / 'smoke.pth'
    save_checkpoint(path, checkpoint(cfg, 'smoke'))
    forbidden = Mock(side_effect=AssertionError('Smoke entered formal trainer'))
    monkeypatch.setattr(training, 'manifest_identity', lambda cfg: {'fixture': 'unit'})
    monkeypatch.setattr(training, 'Trainer', forbidden)
    with pytest.raises(ValueError, match='smoke checkpoint'):
        training.formal_training(cfg, path)
    forbidden.assert_not_called()
