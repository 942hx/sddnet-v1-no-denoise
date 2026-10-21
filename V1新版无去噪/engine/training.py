"""One A0 trainer; small synthetic tests reuse the same update/restore code."""
from __future__ import annotations
import math
import time
import torch
from collections import Counter
from torch.utils.data import DataLoader
from datasets import MILLTripletDataset
from datasets.paired_dataset import ManifestPairedDataset
from datasets.sampling import AnchorSampler
from losses import SDDLoss
from models import SDDNetV1
from models.sddnet_v1 import model_options
from .diagnostics import branch_gradient_norms, diagnostics, fixed_probe
from utils.runtime import seed_everything, choose_device, make_scaler, capture_rng, restore_rng, load_checkpoint, save_checkpoint, project_path, seed_worker
from utils.experiment import ARCH_REVISION, manifest_identity, run_path, training_contract, validate_checkpoint, validate_formal_checkpoint, architecture_hash, object_hash
from utils.io import write_json, write_csv, read_csv
from utils.metrics import psnr_rgb
from losses.ssim import ssim

def optimizer_for(model, cfg):
    """Biases and scalar rho are intentionally excluded from AdamW decay."""
    (decay, no_decay) = ([], [])
    for (name, parameter) in model.named_parameters():
        (no_decay if parameter.ndim <= 1 or name.endswith('.bias') else decay).append(parameter)
    return torch.optim.AdamW([dict(params=decay, weight_decay=cfg['weight_decay']), dict(params=no_decay, weight_decay=0.0)], lr=cfg['learning_rate'])

def accumulation_samples(size, batch_size, steps, batch_index):
    start = batch_index // steps * steps * batch_size
    return min(size, start + steps * batch_size) - start

def concatenate_triplets(batch, device):
    names = ('anchor', 'positive', 'negative')
    low = torch.cat([batch[name + '_low'] for name in names]).to(device, non_blocking=True)
    gt = torch.cat([batch[name + '_gt'] for name in names]).to(device, non_blocking=True)
    levels = torch.cat([batch[name + '_level'] for name in names]).to(device=device, dtype=torch.float32).reshape(-1, 1)
    return (low, gt, levels)

def batch_metadata(batch):
    """Only the loss/sampling audit sees true content and level; never model.forward."""
    if 'anchor_content_id' not in batch:
        return None
    roles = ('anchor', 'positive', 'negative')
    return dict(layout='triplet', content_ids=[s for r in roles for s in batch[r + '_content_id']], group_ids=[s for r in roles for s in batch[r + '_group_id']], raw_levels=[v for r in roles for v in batch[r + '_raw_level'].tolist()])

def window_sample_count(counts, steps, index):
    """Exact normalization for a partial final triplet accumulation window."""
    start = index // steps * steps
    return sum(counts[start:start + steps])

def sampled_amp_update_state(scaler, optimizer):
    """Read scale and unscale_'s overflow flags in ONE existing scale-read sync.

    Only a prospective sampling boundary calls this helper. This replaces that
    update's old get_scale() read (rather than adding another sync), so even a
    skipped candidate does not perform branch traversal or diagnostic transfers.
    GradScaler exposes no public pre-step overflow accessor: these two read-only
    accessors are verified against the supported PyTorch 2.5.1 implementation.
    No scaler state is changed; unsupported implementations fail explicitly.
    """
    if not scaler.is_enabled():
        return (scaler.get_scale(), True)
    if not all((hasattr(scaler, name) for name in ('_get_scale_async', '_found_inf_per_device'))):
        raise RuntimeError('GradScaler pre-step diagnostic accessors are unavailable; use --grad-norm-every 0 or the supported PyTorch version.')
    scale = scaler._get_scale_async()
    flags = list(scaler._found_inf_per_device(optimizer).values())
    if scale is None or not flags:
        raise RuntimeError('Gradient diagnostics require scaler.unscale_(optimizer) before sampling.')
    found_inf = torch.stack([flag.to(device=scale.device) for flag in flags]).sum()
    (old_scale, overflow) = torch.stack((scale.detach(), found_inf)).cpu().tolist()
    return (old_scale, overflow == 0.0)

def log_gradient_samples(writer, samples):
    """TensorBoard steps are successful optimizer updates, never microbatches."""
    for sample in samples:
        for (key, value) in sample['norms'].items():
            if math.isfinite(value):
                writer.add_scalar('gradient/' + key, value, sample['optimizer_step'])

@torch.inference_mode()
def validate(model, dataset, device, max_images=None, details=False):
    """max_images is reserved for the explicitly labelled smoke validation probe."""
    previous = model.training
    model.eval()
    values = []
    per_image = []
    try:
        for index in range(len(dataset) if max_images is None else min(max_images, len(dataset))):
            sample = dataset[index]
            (low, gt) = [sample[key].unsqueeze(0).to(device) for key in ('low', 'gt')]
            outputs = model(low, True) if details else model(low)
            raw = outputs['enhanced']
            if not torch.isfinite(raw).all().item():
                raise FloatingPointError(f"Non-finite validation: {sample['filename']}")
            enhanced = raw.clamp(0, 1)
            values.append((psnr_rgb(enhanced, gt), ssim(enhanced, gt).item()))
            if details:
                metadata = dataset.rows[index]
                predicted = outputs['c_hat'].item()
                gate = 1.0
                delta = (raw - low) / gate
                per_image.append(dict(filename=metadata['filename'], content_id=metadata['content_id'], group_id=metadata.get('group_id', ''), level=float(metadata['level']), split='validation', PSNR_RGB=values[-1][0], SSIM_RGB=values[-1][1], input_PSNR_RGB=psnr_rgb(low, gt), input_SSIM_RGB=ssim(low, gt).item(), delta_psnr_vs_input=values[-1][0] - psnr_rgb(low, gt), delta_psnr_vs_control='NA', c_hat=predicted, illumination_MAE=abs(predicted - (float(metadata['level']) - 1) / 9), gate_mean=gate, delta_abs_mean=delta.abs().mean().item(), gated_delta_abs_mean=(raw - low).abs().mean().item(), rgb_mean_signed_error=(enhanced.mean() - gt.mean()).item(), channel_mean_error_abs=(enhanced.mean((2, 3)) - gt.mean((2, 3))).abs().mean().item(), saturation_fraction=((enhanced <= 0) | (enhanced >= 1)).float().mean().item()))
        result = dict(PSNR_RGB=sum((v[0] for v in values)) / len(values), SSIM_RGB=sum((v[1] for v in values)) / len(values), count=len(values))
        if details:
            from utils.ablation_metrics import paired_summary
            per_level = []
            for level in range(1, 11):
                subset = [r for r in per_image if r['level'] == level]
                row = dict(level=level, count=len(subset), scope='full validation' if max_images is None else 'limited smoke validation')
                for key in ('PSNR_RGB', 'SSIM_RGB', 'input_PSNR_RGB', 'input_SSIM_RGB', 'illumination_MAE', 'gate_mean', 'delta_abs_mean', 'gated_delta_abs_mean', 'rgb_mean_signed_error', 'channel_mean_error_abs', 'saturation_fraction'):
                    row[key] = sum((r[key] for r in subset)) / len(subset) if subset else 'NA'
                row.update(paired_summary(subset))
                per_level.append(row)
            result.update(per_image=per_image, per_level=per_level)
        return result
    finally:
        model.train(previous)

class Trainer:
    """Construct only in training/smoke modes; eval/check-only never creates this class."""

    def __init__(self, cfg, resume=None):
        (self.cfg, self.hashes) = (cfg, manifest_identity(cfg))
        seed_everything(cfg['seed'])
        torch.set_num_threads(cfg.get('cpu_threads', 4))
        self.device = choose_device(cfg.get('device', 'auto'))
        self.use_amp = bool(cfg['amp'] and self.device.type == 'cuda')
        directory = run_path(cfg)
        self.dataset = MILLTripletDataset(project_path(cfg['data']['mill_root']), patch_size=cfg['patch_size'], metadata=directory / 'metadata' / 'mill.csv', illum_min=1, illum_max=10)
        self.validation = ManifestPairedDataset(project_path(cfg['data']['mill_root']), directory / 'metadata' / 'mill.csv', 'validation')
        self.sampler = AnchorSampler(len(self.dataset), cfg['seed'])
        self.loader_generator = torch.Generator().manual_seed(cfg['seed'] + 1)
        options = dict(prefetch_factor=1) if cfg['num_workers'] > 0 else {}
        self.loader = DataLoader(self.dataset, batch_size=cfg['triplet_batch_size'], sampler=self.sampler, num_workers=cfg['num_workers'], pin_memory=cfg['pin_memory'] and self.device.type == 'cuda', persistent_workers=False, worker_init_fn=seed_worker, generator=self.loader_generator, collate_fn=None, **options)
        self.sample_counts = [min(cfg['triplet_batch_size'], len(self.dataset) - i) for i in range(0, len(self.dataset), cfg['triplet_batch_size'])]
        self.model = SDDNetV1(**model_options(cfg)).to(self.device)
        self.criterion = SDDLoss(**cfg['loss']).to(self.device)
        self.optimizer = optimizer_for(self.model, cfg)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=100, eta_min=cfg['min_lr'])
        self.scaler = make_scaler(self.use_amp)
        (self.completed_epoch, self.global_step, self.optimizer_updates, self.overflows, self.attempted_updates) = (0, 0, 0, 0, 0)
        (self.best_metric, self.best_epoch, self.peak_memory, self.elapsed_seconds) = (-math.inf, 0, 0, 0.0)
        self.last_gradient_norms = {}
        self.last_gradient_step = None
        self.gradient_samples = []
        self.grad_norm_every = cfg.get('diagnostics', {}).get('grad_norm_every', 100)
        if type(self.grad_norm_every) is not int or self.grad_norm_every < 0:
            raise ValueError('grad_norm_every must be a nonnegative integer.')
        self.logs = []
        self.image_exposures = 0
        self.last_coverage = {}
        self.triplet_rows = []
        if resume:
            self.restore(resume)

    def restore(self, path):
        state = load_checkpoint(path)
        validate_checkpoint(state, self.cfg, self.hashes)
        if type(state.get('optimizer_updates')) is not int or state['optimizer_updates'] < 0:
            raise ValueError('Resume requires a reliable nonnegative optimizer_updates counter; it cannot be inferred from microbatch global_step.')
        self.model.load_state_dict(state['model'], strict=True)
        self.optimizer.load_state_dict(state['optimizer'])
        self.scheduler.load_state_dict(state['scheduler'])
        self.scaler.load_state_dict(state['scaler'])
        self.sampler.load_state_dict(state['sampler'])
        self.loader_generator.set_state(state['loader_generator'])
        for key in ('completed_epoch', 'global_step', 'optimizer_updates', 'overflows', 'attempted_updates', 'best_metric', 'best_epoch', 'peak_memory', 'elapsed_seconds'):
            setattr(self, key, state[key])
        restore_rng(state['rng'])
        self.image_exposures = state.get('image_exposures', self.global_step * self.cfg['triplet_batch_size'] * 3)

    def checkpoint(self):
        state = dict(arch_revision=ARCH_REVISION, model=self.model.state_dict(), optimizer=self.optimizer.state_dict(), scheduler=self.scheduler.state_dict(), scaler=self.scaler.state_dict(), completed_epoch=self.completed_epoch, global_step=self.global_step, optimizer_updates=self.optimizer_updates, overflows=self.overflows, attempted_updates=self.attempted_updates, best_metric=self.best_metric, best_epoch=self.best_epoch, peak_memory=self.peak_memory, elapsed_seconds=self.elapsed_seconds, config=self.cfg, contract=training_contract(self.cfg), manifest_hashes=self.hashes, rng=capture_rng(), sampler=self.sampler.state_dict(), loader_generator=self.loader_generator.get_state())
        if self.cfg.get('ablation_version'):
            state.update(architecture_hash=architecture_hash(self.cfg), config_hash=object_hash(training_contract(self.cfg)), variant=self.cfg['variant'], seed=self.cfg['seed'], image_exposures=self.image_exposures, scheduler_step_unit='completed_exposure_epoch', scheduler_optimizer_update=self.optimizer_updates)
        return state

    def _fail(self, message, batch, values):
        write_json(run_path(self.cfg) / 'failure.json', dict(reason=message, global_step=self.global_step, filenames={'images': batch['filenames'], 'content_ids': batch['content_ids']} if 'filenames' in batch else {key: value for (key, value) in batch.items() if key.endswith('_filename')}, statistics=values))
        raise FloatingPointError(message)

    def epoch(self, epoch, max_updates=None):
        """max_updates is used only by the smoke caller; formal caller always passes None."""
        cfg = self.cfg
        self.logs = []
        self.triplet_rows = []
        self.last_gradient_norms = {}
        self.last_gradient_step = None
        self.gradient_samples = []
        self.sampler.start_epoch(epoch)
        start_batch = self.sampler.position // cfg['triplet_batch_size']
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        (sums, count, initial_updates) = ({}, 0, self.optimizer_updates)
        if self.device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(self.device)
        started = time.perf_counter()
        if self.device.type == 'cuda':
            torch.cuda.synchronize(self.device)
        started = time.perf_counter()
        (coverage, levels_seen, filenames) = (Counter(), Counter(), Counter())
        epoch_images = 0
        for (offset, batch) in enumerate(self.loader):
            step = start_batch + offset
            (low, gt, levels) = concatenate_triplets(batch, self.device)
            info = batch_metadata(batch)
            if info:
                coverage.update(info['content_ids'])
                levels_seen.update((str(int(v)) for v in info['raw_levels']))
                names = [s for role in ('anchor', 'positive', 'negative') for s in batch[role + '_filename']]
                filenames.update(names)
            stats = {}
            for (name, value) in (('input', low), ('gt', gt), ('levels', levels)):
                stats[name] = dict(min=value.detach().min().item(), max=value.detach().max().item())
                if not torch.isfinite(value).all().item() or stats[name]['min'] < 0 or stats[name]['max'] > 1:
                    self._fail('Invalid/non-finite training input or label', batch, stats)
            with torch.autocast(device_type=self.device.type, enabled=self.use_amp):
                output = self.model(low, return_latents=True)
                if info:
                    output['_batch_info'] = info
                losses = self.criterion(output, gt, levels, progress=self.global_step / len(self.loader))
            if not torch.isfinite(torch.stack(list(losses.values()))).all().item():
                self._fail('Non-finite loss', batch, dict(stats, losses={k: v.detach().item() for (k, v) in losses.items()}))
            triplets = batch['anchor_low'].shape[0]
            denominator = window_sample_count(self.sample_counts, cfg['accumulation_steps'], step)
            self.scaler.scale(losses['total'] * triplets / denominator).backward()
            boundary = (step + 1) % cfg['accumulation_steps'] == 0 or step + 1 == len(self.loader)
            (norm, overflow) = (None, False)
            sampled_norms = None
            if boundary:
                self.scaler.unscale_(self.optimizer)
                sample_due = self.grad_norm_every > 0 and (self.optimizer_updates + 1) % self.grad_norm_every == 0
                old_scale = None
                if sample_due:
                    (old_scale, will_update) = sampled_amp_update_state(self.scaler, self.optimizer)
                    if will_update:
                        sampled_norms = branch_gradient_norms(self.model)
                norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg['grad_clip_norm']).item()
                if not self.use_amp and (not math.isfinite(norm)):
                    self._fail('Non-finite FP32 gradient norm', batch, dict(stats, grad_norm=norm))
                if old_scale is None:
                    old_scale = self.scaler.get_scale()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                overflow = self.scaler.get_scale() < old_scale
                self.overflows += int(overflow)
                self.optimizer_updates += int(not overflow)
                self.attempted_updates += 1
                if not overflow and sampled_norms is not None:
                    self.last_gradient_norms = sampled_norms
                    self.last_gradient_step = self.optimizer_updates
                    self.gradient_samples.append(dict(optimizer_step=self.optimizer_updates, norms=sampled_norms))
                self.optimizer.zero_grad(set_to_none=True)
                if self.attempted_updates >= 10 and self.overflows / self.attempted_updates > 0.2:
                    self._fail('Frequent AMP overflows: training is not numerically accepted', batch, dict(stats, overflows=self.overflows))
            values = {key: value.detach().item() for (key, value) in losses.items()}
            if info and getattr(self.criterion, 'last_triplet_details', None) is not None:
                for (a, p, n, valid, dap, dan) in self.criterion.last_triplet_details:
                    (a, p, n) = (int(a), int(p), int(n))
                    legal = [j for (j, s) in enumerate(info['content_ids']) if s != info['content_ids'][a] and (not (info['group_ids'][a] and info['group_ids'][a] == info['group_ids'][j]))]
                    preferred = [j for j in legal if info['raw_levels'][j] == info['raw_levels'][a]]
                    category = self.criterion.mining_mode
                    self.triplet_rows.append(dict(epoch=epoch, microbatch_step=self.global_step + 1, optimizer_updates=self.optimizer_updates, anchor_filename=names[a], positive_filename=names[p], negative_filename=names[n] if valid else '', content_id=info['content_ids'][a], negative_content_id=info['content_ids'][n] if valid else '', level=info['raw_levels'][a], negative_level=info['raw_levels'][n] if valid else '', valid=bool(valid), d_ap=dap, d_an=dan if valid else 'NA', active=bool(dap - dan + self.criterion.triplet.margin > 0) if valid else False, valid_negative_count=len(preferred or legal), selection_category=category if valid else 'skip_no_legal'))
            for (key, value) in values.items():
                sums[key] = sums.get(key, 0.0) + triplets * value
            self.global_step += 1
            count += triplets
            epoch_images += len(low)
            self.image_exposures += len(low)
            row = dict(epoch=epoch, batch=step + 1, global_step=self.global_step, optimizer_updates=self.optimizer_updates, grad_norm_preclip=norm, AMP_overflow=overflow, branch_grad_norm_sampled=sampled_norms is not None and (not overflow), **values)
            if cfg.get('ablation_version'):
                row.update(physical_microbatch_images=len(low), gradient_accumulation_steps=cfg['accumulation_steps'], effective_images_this_window=denominator * 3, image_exposures=self.image_exposures, unique_contents_this_epoch=len(coverage), scheduler_last_epoch=self.scheduler.last_epoch, lr_used=self.optimizer.param_groups[0]['lr'])
            if sampled_norms is not None and (not overflow):
                row.update(sampled_norms)
            self.logs.append(row)
            if boundary:
                print('UPDATE ' + str(self.optimizer_updates) + ' ' + ' '.join((f'{k}={v:.6g}' for (k, v) in values.items())) + f' grad_norm_preclip={norm:.6g} overflow={overflow}', flush=True)
            del output, losses, low, gt, levels
            if max_updates is not None and self.optimizer_updates - initial_updates >= max_updates:
                break
        if self.device.type == 'cuda':
            torch.cuda.synchronize(self.device)
        self.last_epoch_seconds = time.perf_counter() - started
        self.elapsed_seconds += self.last_epoch_seconds
        self.last_coverage = dict(image_exposures=epoch_images, unique_contents=len(coverage), unique_low_files=len(filenames), repeated_file_exposures=epoch_images - len(filenames), content_exposures=dict(coverage), level_exposures=dict(levels_seen), file_exposures=dict(filenames), epoch_definition='every low once as anchor; positives/negatives may repeat')
        if self.device.type == 'cuda':
            self.peak_memory = max(self.peak_memory, torch.cuda.max_memory_allocated(self.device))
        expected = sum(self.sample_counts)
        if max_updates is None and (count != expected or self.sampler.position != len(self.dataset)):
            raise RuntimeError('Incomplete formal epoch; it will not be marked completed.')
        return ({key: value / count for (key, value) in sums.items()}, count)

    def gradient_epoch_metrics(self):
        """Means include only genuine samples; absent norms never become zeros."""
        result = dict(grad_norm_samples=len(self.gradient_samples))
        for key in self.last_gradient_norms:
            values = [sample['norms'][key] for sample in self.gradient_samples if math.isfinite(sample['norms'][key])]
            result[key + '_mean'] = sum(values) / len(values) if values else 'NA'
        return result

    def diagnose(self):
        (low, gt, labels) = fixed_probe(self.validation, self.cfg['diagnostics']['patch_size'])
        median = torch.tensor([(record.level - 1) / 9 for record in self.dataset.records]).median().item()
        result = diagnostics(self.model, low.to(self.device), gt.to(self.device), labels.to(self.device), self.criterion, median, self.last_gradient_norms)
        result.update(self.gradient_epoch_metrics())
        if self.last_gradient_step is not None:
            result['branch_grad_norm_optimizer_step'] = self.last_gradient_step
        return result

def formal_training(cfg, resume=None):
    directory = run_path(cfg)
    checkpoints = directory / 'checkpoints'
    if resume:
        validate_formal_checkpoint(load_checkpoint(resume), cfg, manifest_identity(cfg), for_resume=True)
    if not torch.cuda.is_available() or choose_device(cfg['device']).type != 'cuda':
        raise RuntimeError('Formal 100-epoch training requires CUDA; CPU checks only.')
    if not resume and ((directory / 'history.csv').exists() or list(checkpoints.glob('*.pth'))):
        raise RuntimeError('Run already has training results. Use --resume or a new run_dir.')
    trainer = Trainer(cfg, resume)
    history = [row for row in read_csv(directory / 'history.csv') if int(row['completed_epoch']) <= trainer.completed_epoch]
    diagnostic_rows = [row for row in read_csv(directory / 'diagnostics.csv') if int(row['completed_epoch']) <= trainer.completed_epoch]
    if resume and trainer.completed_epoch and (trainer.completed_epoch not in {int(r['completed_epoch']) for r in history}):
        import json
        journal = directory / 'epoch_records' / f'epoch_{trainer.completed_epoch:04d}.json'
        if journal.exists():
            recovered = json.loads(journal.read_text(encoding='utf-8'))
        else:
            rng = capture_rng()
            try:
                scores = validate(trainer.model, trainer.validation, trainer.device)
            finally:
                restore_rng(rng)
            unavailable = ('total', 'rec', 'l1', 'ssim_loss', 'scene', 'illumination', 'cons', 'weighted_ssim', 'weighted_scene', 'weighted_illum', 'weighted_cons', 'lambda_cons')
            row = dict(completed_epoch=trainer.completed_epoch, anchors=sum(trainer.sample_counts), optimizer_updates=trainer.optimizer_updates, val_PSNR_RGB=scores['PSNR_RGB'], val_SSIM_RGB=scores['SSIM_RGB'], lr='NA', peak_train_bytes=trainer.peak_memory, **{key: 'NA' for key in unavailable})
            row['recovery_note'] = 'Validation recomputed from saved full-epoch weights; training means and epoch training LR unavailable. Optimizer stores next-epoch LR.'
            recovered = dict(history=row, diagnostics=None)
            write_json(journal, recovered)
            write_json(directory / 'log_recovery.json', dict(completed_epoch=trainer.completed_epoch, checkpoint=str(resume), reason='Original history.csv write failed after full checkpoint save.', validation='Recomputed, full 120-image validation, unchanged checkpoint weights.', unavailable='Per-epoch training means and original diagnostics for this epoch are NA; no simulated values.', checkpoint_updates_repeated=False))
        history.append(recovered['history'])
        if recovered.get('diagnostics') is not None:
            diagnostic_rows.append(recovered['diagnostics'])
        write_csv(directory / 'history.csv', history, allow_locked=True)
        write_csv(directory / 'diagnostics.csv', diagnostic_rows, allow_locked=True)
        print(f'RECOVERED complete checkpoint epoch {trainer.completed_epoch}; next epoch {trainer.completed_epoch + 1}.', flush=True)
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter(str(directory / 'tensorboard'))
    try:
        for epoch in range(trainer.completed_epoch + 1, 101):
            epoch_started = time.perf_counter()
            epoch_train_lrs = tuple((group['lr'] for group in trainer.optimizer.param_groups))
            epoch_train_lr = epoch_train_lrs[0]
            (losses, count) = trainer.epoch(epoch)
            scores = validate(trainer.model, trainer.validation, trainer.device, details=True) if cfg.get('ablation_version') else validate(trainer.model, trainer.validation, trainer.device)
            diagnostic = trainer.diagnose()
            if cfg.get('ablation_version'):
                write_csv(directory / 'validation' / f'epoch_{epoch:04d}_per_image.csv', scores['per_image'])
                write_csv(directory / 'validation' / f'epoch_{epoch:04d}_per_level.csv', scores['per_level'])
                if cfg['diagnostics'].get('condition_every_epoch', True):
                    from .condition_diagnostics import run_condition_diagnostics
                    run_condition_diagnostics(trainer.model, trainer.validation, directory / 'conditions' / f'epoch_{epoch:04d}', size=cfg['diagnostics']['patch_size'], contents=cfg['diagnostics'].get('condition_contents', 3), seed=cfg['seed'], provenance=dict(variant=cfg['variant'], epoch=epoch, optimizer_updates=trainer.optimizer_updates))
            trainer.completed_epoch = epoch
            trainer.scheduler.step()
            improved = scores['PSNR_RGB'] > trainer.best_metric
            if improved:
                (trainer.best_metric, trainer.best_epoch) = (scores['PSNR_RGB'], epoch)
            row = dict(completed_epoch=epoch, anchors=count, optimizer_updates=trainer.optimizer_updates, val_PSNR_RGB=scores['PSNR_RGB'], val_SSIM_RGB=scores['SSIM_RGB'], lr=epoch_train_lr, peak_train_bytes=trainer.peak_memory, **losses, **{f'lr_group_{index}': lr for (index, lr) in enumerate(epoch_train_lrs)}, **trainer.gradient_epoch_metrics())
            if cfg.get('ablation_version'):
                row.update(train_loop_seconds=trainer.last_epoch_seconds, epoch_with_validation_seconds=time.perf_counter() - epoch_started, epoch_image_exposures=trainer.last_coverage['image_exposures'], image_exposures=trainer.image_exposures, unique_contents=trainer.last_coverage['unique_contents'], unique_low_files=trainer.last_coverage['unique_low_files'], attempted_updates=trainer.attempted_updates, AMP_overflows=trainer.overflows, scheduler_last_epoch=trainer.scheduler.last_epoch, scheduler_optimizer_update=trainer.optimizer_updates)
                write_json(directory / 'coverage' / f'epoch_{epoch:04d}.json', trainer.last_coverage)
                write_csv(directory / 'mining' / f'epoch_{epoch:04d}_triplets.csv', trainer.triplet_rows)
                from utils.ablation_metrics import mining_summary
                row.update(mining_summary(trainer.triplet_rows))
            history.append(row)
            diagnostic_row = dict(completed_epoch=epoch, **diagnostic)
            diagnostic_rows.append(diagnostic_row)
            write_json(directory / 'epoch_records' / f'epoch_{epoch:04d}.json', dict(history=row, diagnostics=diagnostic_row))
            state = trainer.checkpoint()
            state['run_kind'] = 'formal'
            if improved:
                save_checkpoint(checkpoints / 'best.pth', state)
            save_checkpoint(checkpoints / 'last.pth', state)
            write_csv(directory / 'history.csv', history, allow_locked=True)
            write_csv(directory / 'diagnostics.csv', diagnostic_rows, allow_locked=True)
            for (key, value) in row.items():
                if isinstance(value, (int, float)) and math.isfinite(value):
                    writer.add_scalar('epoch/' + key, value, epoch)
            log_gradient_samples(writer, trainer.gradient_samples)
            writer.flush()
            print(f'EPOCH {epoch}/100 COMPLETE ' + str(row), flush=True)
            from .reporting import report
            report(cfg, stage='formal_training', trainer=trainer)
    finally:
        writer.close()
    return checkpoints / 'best.pth'

def validate_best_checkpoint(cfg, checkpoint_path):
    """Export THIS run's strict-loaded best on the complete validation manifest.

    No optimizer, final-data audit, test dataset, LPIPS or profile is constructed.
    This post-training export does not change any training/update/scheduler path.
    """
    from utils.experiment import file_hash, read_json
    directory = run_path(cfg)
    expected = (directory / 'checkpoints/best.pth').resolve()
    checkpoint_path = project_path(checkpoint_path).resolve()
    if checkpoint_path != expected:
        raise ValueError("Validation-best export must load this run's checkpoints/best.pth.")
    hashes = manifest_identity(cfg)
    state = load_checkpoint(checkpoint_path)
    validate_formal_checkpoint(state, cfg, hashes)
    if state['completed_epoch'] != state['best_epoch']:
        raise ValueError('best.pth does not identify its own validation-best epoch.')
    checkpoint_hash = file_hash(checkpoint_path)
    destination = directory / 'validation/best'
    if destination.exists() and any(destination.iterdir()):
        provenance = read_json(destination / 'provenance.json')
        if provenance['checkpoint_sha256'] != checkpoint_hash or provenance['manifest_hashes'] != hashes or any((file_hash(destination / name) != sha for (name, sha) in provenance['csv_sha256'].items())):
            raise ValueError('Existing best validation evidence differs; preserve it and investigate.')
        return read_json(destination / 'summary.json')
    dataset = ManifestPairedDataset(project_path(cfg['data']['mill_root']), directory / 'metadata/mill.csv', 'validation')
    audit = read_json(directory / 'data_audit.json')
    expected_count = audit['splits']['mill/validation']['valid_pairs']
    if len(dataset) != expected_count:
        raise ValueError('Validation count differs from the complete audited split.')
    device = choose_device(cfg['device'])
    model = SDDNetV1(**model_options(cfg)).to(device).float().eval()
    model.load_state_dict(state['model'], strict=True)
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    scores = validate(model, dataset, device, details=True)
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    seconds = time.perf_counter() - started
    if scores['count'] != expected_count or len(scores['per_image']) != expected_count:
        raise ValueError('Best validation export is incomplete.')
    if not math.isclose(scores['PSNR_RGB'], state['best_metric'], rel_tol=0, abs_tol=1e-05):
        raise ValueError('Recomputed full validation PSNR differs from best checkpoint selection metric.')
    images = [dict(r, dataset='mill', split='validation', checkpoint_epoch=state['completed_epoch'], checkpoint_sha256=checkpoint_hash) for r in scores['per_image']]
    levels = [dict(r, dataset='mill', split='validation') for r in scores['per_level']]
    summary = dict(dataset='mill', split='validation', view='full_native_resolution', count=expected_count, PSNR_RGB=scores['PSNR_RGB'], SSIM_RGB=scores['SSIM_RGB'], validation_seconds=seconds, best_epoch=state['completed_epoch'], checkpoint_sha256=checkpoint_hash, selection_rule='mean of per-image full MILL validation PSNR_RGB', protocol='FP32 batch1 RGB; clamp outside forward; float predictions before PNG quantization; no resize/GT alignment/tiles/crop; SSIM win11 sigma1.5 data_range1', config_hash=object_hash(training_contract(cfg)), manifest_hashes=hashes, optimizer_updates_at_best=state['optimizer_updates'])
    for key in ('input_PSNR_RGB', 'input_SSIM_RGB', 'illumination_MAE', 'c_hat', 'delta_psnr_vs_input'):
        summary[key] = sum((r[key] for r in images)) / len(images)
    from utils.ablation_metrics import paired_summary
    summary.update(paired_summary(images, seed=cfg['seed']))
    write_csv(destination / 'per_image.csv', images)
    write_csv(destination / 'per_level.csv', levels)
    write_csv(destination / 'summary.csv', [summary])
    write_json(destination / 'summary.json', summary)
    write_json(destination / 'provenance.json', dict(checkpoint=str(checkpoint_path), checkpoint_sha256=checkpoint_hash, best_epoch=state['completed_epoch'], run_kind=state['run_kind'], variant=cfg.get('variant', 'historical'), config_hash=summary['config_hash'], resolved_config_sha256=file_hash(directory / 'config_resolved.yaml'), manifest_hashes=hashes, complete_validation_count=expected_count, selection_rule=summary['selection_rule'], csv_sha256={name: file_hash(destination / name) for name in ('per_image.csv', 'per_level.csv', 'summary.csv')}))
    print('BEST FULL VALIDATION ' + str(summary), flush=True)
    return summary
