"""One trainer for formal full epochs and independent 20-update real-data smoke tests."""
from __future__ import annotations
import math
import time
import torch
from torch.utils.data import DataLoader
from datasets import MILLTripletDataset
from datasets.paired_dataset import ManifestPairedDataset
from datasets.sampling import AnchorSampler
from losses import SDDLoss
from models import SDDNetV1
from .diagnostics import branch_gradient_norms, diagnostics, fixed_probe
from utils.runtime import (seed_everything, choose_device, make_scaler, capture_rng, restore_rng,
                           load_checkpoint, save_checkpoint, project_path, seed_worker)
from utils.experiment import ARCH_REVISION, manifest_identity, run_path, training_contract, validate_checkpoint
from utils.io import write_json, write_csv, read_csv
from utils.metrics import psnr_rgb
from losses.ssim import ssim


def optimizer_for(model, cfg):
    """Biases and scalar rho are intentionally excluded from AdamW decay."""
    decay, no_decay = [], []
    for name, parameter in model.named_parameters():
        (no_decay if parameter.ndim <= 1 or name.endswith('.bias') else decay).append(parameter)
    return torch.optim.AdamW([dict(params=decay, weight_decay=cfg['weight_decay']),
                              dict(params=no_decay, weight_decay=0.0)], lr=cfg['learning_rate'])


def accumulation_samples(size, batch_size, steps, batch_index):
    start = batch_index // steps * steps * batch_size
    return min(size, start + steps * batch_size) - start


def concatenate_triplets(batch, device):
    names = ('anchor', 'positive', 'negative')
    low = torch.cat([batch[name+'_low'] for name in names]).to(device, non_blocking=True)
    gt = torch.cat([batch[name+'_gt'] for name in names]).to(device, non_blocking=True)
    levels = torch.cat([batch[name+'_level'] for name in names]).to(device=device, dtype=torch.float32).reshape(-1,1)
    return low, gt, levels


@torch.inference_mode()
def validate(model, dataset, device, max_images=None):
    """max_images is reserved for the explicitly labelled smoke validation probe."""
    previous = model.training
    model.eval()
    values = []
    try:
        for index in range(len(dataset) if max_images is None else min(max_images,len(dataset))):
            sample = dataset[index]
            low, gt = [sample[key].unsqueeze(0).to(device) for key in ('low','gt')]
            raw = model(low)['enhanced']
            if not torch.isfinite(raw).all().item():
                raise FloatingPointError(f'Non-finite validation: {sample["filename"]}')
            enhanced = raw.clamp(0,1)
            values.append((psnr_rgb(enhanced,gt), ssim(enhanced,gt).item()))
        return dict(PSNR_RGB=sum(v[0] for v in values)/len(values),
                    SSIM_RGB=sum(v[1] for v in values)/len(values), count=len(values))
    finally:
        model.train(previous)


class Trainer:
    """Construct only in training/smoke modes; eval/check-only never creates this class."""
    def __init__(self, cfg, resume=None):
        self.cfg, self.hashes = cfg, manifest_identity(cfg)
        seed_everything(cfg['seed'])
        torch.set_num_threads(cfg.get('cpu_threads',4))
        self.device = choose_device(cfg.get('device','auto'))
        self.use_amp = bool(cfg['amp'] and self.device.type == 'cuda')
        directory = run_path(cfg)
        self.dataset = MILLTripletDataset(project_path(cfg['data']['mill_root']), patch_size=cfg['patch_size'],
                                          metadata=directory/'metadata'/'mill.csv', illum_min=1, illum_max=10)
        self.validation = ManifestPairedDataset(project_path(cfg['data']['mill_root']), directory/'metadata'/'mill.csv','validation')
        self.sampler = AnchorSampler(len(self.dataset),cfg['seed'])
        self.loader_generator = torch.Generator().manual_seed(cfg['seed']+1)
        options = dict(prefetch_factor=1) if cfg['num_workers'] > 0 else {}
        self.loader = DataLoader(self.dataset,batch_size=cfg['triplet_batch_size'],sampler=self.sampler,
            num_workers=cfg['num_workers'],pin_memory=cfg['pin_memory'] and self.device.type=='cuda',
            persistent_workers=False,worker_init_fn=seed_worker,generator=self.loader_generator,**options)
        self.model = SDDNetV1().to(self.device)
        self.criterion = SDDLoss(**cfg['loss']).to(self.device)
        self.optimizer = optimizer_for(self.model,cfg)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer,T_max=100,eta_min=cfg['min_lr'])
        self.scaler = make_scaler(self.use_amp)
        self.completed_epoch, self.global_step, self.optimizer_updates, self.overflows, self.attempted_updates = 0,0,0,0,0
        self.best_metric, self.best_epoch, self.peak_memory, self.elapsed_seconds = -math.inf,0,0,0.0
        self.last_gradient_norms = {}
        self.logs = []
        if resume:
            self.restore(resume)

    def restore(self, path):
        state = load_checkpoint(path)
        validate_checkpoint(state,self.cfg,self.hashes)
        self.model.load_state_dict(state['model'],strict=True)
        self.optimizer.load_state_dict(state['optimizer'])
        self.scheduler.load_state_dict(state['scheduler'])
        self.scaler.load_state_dict(state['scaler'])
        self.sampler.load_state_dict(state['sampler'])
        self.loader_generator.set_state(state['loader_generator'])
        for key in ('completed_epoch','global_step','optimizer_updates','overflows','attempted_updates',
                    'best_metric','best_epoch','peak_memory','elapsed_seconds'):
            setattr(self,key,state[key])
        restore_rng(state['rng'])

    def checkpoint(self):
        return dict(arch_revision=ARCH_REVISION,model=self.model.state_dict(),optimizer=self.optimizer.state_dict(),
            scheduler=self.scheduler.state_dict(),scaler=self.scaler.state_dict(),completed_epoch=self.completed_epoch,
            global_step=self.global_step,optimizer_updates=self.optimizer_updates,overflows=self.overflows,
            attempted_updates=self.attempted_updates,best_metric=self.best_metric,best_epoch=self.best_epoch,
            peak_memory=self.peak_memory,elapsed_seconds=self.elapsed_seconds,config=self.cfg,
            contract=training_contract(self.cfg),manifest_hashes=self.hashes,rng=capture_rng(),
            sampler=self.sampler.state_dict(),loader_generator=self.loader_generator.get_state())

    def _fail(self, message, batch, values):
        write_json(run_path(self.cfg)/'failure.json',dict(reason=message,global_step=self.global_step,
            filenames={key:value for key,value in batch.items() if key.endswith('_filename')},statistics=values))
        raise FloatingPointError(message)

    def epoch(self, epoch, max_updates=None):
        """max_updates is used only by the smoke caller; formal caller always passes None."""
        cfg = self.cfg
        self.logs = []
        self.sampler.start_epoch(epoch)
        start_batch = self.sampler.position // cfg['triplet_batch_size']
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        sums,count,initial_updates = {},0,self.optimizer_updates
        if self.device.type=='cuda':
            torch.cuda.reset_peak_memory_stats(self.device)
        started = time.perf_counter()
        for offset,batch in enumerate(self.loader):
            step = start_batch+offset
            low,gt,levels = concatenate_triplets(batch,self.device)
            stats = {}
            for name,value in (('input',low),('gt',gt),('levels',levels)):
                stats[name] = dict(min=value.detach().min().item(),max=value.detach().max().item())
                if not torch.isfinite(value).all().item() or stats[name]['min']<0 or stats[name]['max']>1:
                    self._fail('Invalid/non-finite training input or label',batch,stats)
            with torch.autocast(device_type=self.device.type,enabled=self.use_amp):
                output = self.model(low,return_latents=True)
                losses = self.criterion(output,gt,levels,progress=self.global_step/len(self.loader))
            if not torch.isfinite(torch.stack(list(losses.values()))).all().item():
                self._fail('Non-finite loss',batch,dict(stats,losses={k:v.detach().item() for k,v in losses.items()}))
            triplets = batch['anchor_low'].shape[0]
            denominator = accumulation_samples(len(self.dataset),cfg['triplet_batch_size'],cfg['accumulation_steps'],step)
            self.scaler.scale(losses['total']*triplets/denominator).backward()
            boundary = (step+1)%cfg['accumulation_steps']==0 or step+1==len(self.loader)
            norm,overflow = None,False
            if boundary:
                self.scaler.unscale_(self.optimizer)
                self.last_gradient_norms = branch_gradient_norms(self.model)
                norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(),cfg['grad_clip_norm']).item()
                if not self.use_amp and not math.isfinite(norm):
                    self._fail('Non-finite FP32 gradient norm',batch,dict(stats,grad_norm=norm))
                old_scale = self.scaler.get_scale()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                overflow = self.scaler.get_scale()<old_scale
                self.overflows += int(overflow)
                self.optimizer_updates += int(not overflow)
                self.attempted_updates += 1
                self.optimizer.zero_grad(set_to_none=True)
                if self.attempted_updates >= 10 and self.overflows/self.attempted_updates > .2:
                    self._fail('Frequent AMP overflows: training is not numerically accepted',batch,dict(stats,overflows=self.overflows))
            values = {key:value.detach().item() for key,value in losses.items()}
            for key,value in values.items():
                sums[key] = sums.get(key,0.0)+triplets*value
            self.global_step += 1
            count += triplets
            row = dict(epoch=epoch,batch=step+1,global_step=self.global_step,optimizer_updates=self.optimizer_updates,
                       grad_norm_preclip=norm,AMP_overflow=overflow,**values)
            self.logs.append(row)
            if boundary:
                print('UPDATE '+str(self.optimizer_updates)+' '+ ' '.join(f'{k}={v:.6g}' for k,v in values.items())+
                      f' grad_norm_preclip={norm:.6g} overflow={overflow}',flush=True)
            del output,losses,low,gt,levels
            if max_updates is not None and self.optimizer_updates-initial_updates>=max_updates:
                break
        self.elapsed_seconds += time.perf_counter()-started
        if self.device.type=='cuda':
            self.peak_memory = max(self.peak_memory,torch.cuda.max_memory_allocated(self.device))
        if max_updates is None and (count!=len(self.dataset) or self.sampler.position!=len(self.dataset)):
            raise RuntimeError('Incomplete formal epoch; it will not be marked completed.')
        return {key:value/count for key,value in sums.items()},count

    def diagnose(self):
        low,gt,labels = fixed_probe(self.validation,self.cfg['diagnostics']['patch_size'])
        median = torch.tensor([(record.level-1)/9 for record in self.dataset.records]).median().item()
        return diagnostics(self.model,low.to(self.device),gt.to(self.device),labels.to(self.device),self.criterion,
                           median,self.last_gradient_norms)


def formal_training(cfg,resume=None):
    directory = run_path(cfg)
    checkpoints = directory/'checkpoints'
    if not torch.cuda.is_available() or choose_device(cfg['device']).type!='cuda':
        raise RuntimeError('Formal 100-epoch training requires CUDA; CPU checks only.')
    if not resume and ((directory/'history.csv').exists() or list(checkpoints.glob('*.pth'))):
        raise RuntimeError('Run already has training results. Use --resume or a new run_dir.')
    trainer = Trainer(cfg,resume)
    history = [row for row in read_csv(directory/'history.csv') if int(row['completed_epoch'])<=trainer.completed_epoch]
    diagnostic_rows = [row for row in read_csv(directory/'diagnostics.csv') if int(row['completed_epoch'])<=trainer.completed_epoch]
    # The checkpoint is authoritative if a legacy CSV write failed after saving
    # a full epoch. Never replay those optimizer updates or invent missing loss.
    if resume and trainer.completed_epoch and trainer.completed_epoch not in {int(r['completed_epoch']) for r in history}:
        import json
        journal = directory/'epoch_records'/f'epoch_{trainer.completed_epoch:04d}.json'
        if journal.exists():
            recovered = json.loads(journal.read_text(encoding='utf-8'))
        else:
            rng = capture_rng()
            try:
                scores = validate(trainer.model,trainer.validation,trainer.device)
            finally:
                restore_rng(rng)
            unavailable = ('total','rec','l1','ssim_loss','scene','illumination','cons',
                           'weighted_ssim','weighted_scene','weighted_illum','weighted_cons','lambda_cons')
            row = dict(completed_epoch=trainer.completed_epoch,anchors=len(trainer.dataset),
                       optimizer_updates=trainer.optimizer_updates,val_PSNR_RGB=scores['PSNR_RGB'],
                       val_SSIM_RGB=scores['SSIM_RGB'],lr=trainer.optimizer.param_groups[0]['lr'],
                       peak_train_bytes=trainer.peak_memory,**{key:'NA' for key in unavailable})
            row['recovery_note']='Validation recomputed from saved full-epoch weights; training means unavailable after original CSV write failed.'
            recovered = dict(history=row,diagnostics=None)
            write_json(journal,recovered)
            write_json(directory/'log_recovery.json',dict(completed_epoch=trainer.completed_epoch,
                checkpoint=str(resume),reason='Original history.csv write failed after full checkpoint save.',
                validation='Recomputed, full 120-image validation, unchanged checkpoint weights.',
                unavailable='Per-epoch training means and original diagnostics for this epoch are NA; no simulated values.',
                checkpoint_updates_repeated=False))
        history.append(recovered['history'])
        if recovered.get('diagnostics') is not None:
            diagnostic_rows.append(recovered['diagnostics'])
        write_csv(directory/'history.csv',history,allow_locked=True)
        write_csv(directory/'diagnostics.csv',diagnostic_rows,allow_locked=True)
        print(f'RECOVERED complete checkpoint epoch {trainer.completed_epoch}; next epoch {trainer.completed_epoch+1}.',flush=True)
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter(str(directory/'tensorboard'))
    try:
        for epoch in range(trainer.completed_epoch+1,101):
            losses,count = trainer.epoch(epoch)
            scores = validate(trainer.model,trainer.validation,trainer.device)
            diagnostic = trainer.diagnose()
            trainer.completed_epoch = epoch
            trainer.scheduler.step()
            improved = scores['PSNR_RGB']>trainer.best_metric
            if improved:
                trainer.best_metric,trainer.best_epoch = scores['PSNR_RGB'],epoch
            row = dict(completed_epoch=epoch,anchors=count,optimizer_updates=trainer.optimizer_updates,
                       val_PSNR_RGB=scores['PSNR_RGB'],val_SSIM_RGB=scores['SSIM_RGB'],
                       lr=trainer.optimizer.param_groups[0]['lr'],peak_train_bytes=trainer.peak_memory,**losses)
            history.append(row)
            diagnostic_row = dict(completed_epoch=epoch,**diagnostic)
            diagnostic_rows.append(diagnostic_row)
            # Journal complete measurements before checkpoint/CSV commits, so a
            # locked spreadsheet cannot discard the only copy of an epoch row.
            write_json(directory/'epoch_records'/f'epoch_{epoch:04d}.json',
                       dict(history=row,diagnostics=diagnostic_row))
            state = trainer.checkpoint()
            state['run_kind'] = 'formal'
            if improved:
                save_checkpoint(checkpoints/'best.pth',state)
            save_checkpoint(checkpoints/'last.pth',state)
            write_csv(directory/'history.csv',history,allow_locked=True)
            write_csv(directory/'diagnostics.csv',diagnostic_rows,allow_locked=True)
            for key,value in row.items():
                if isinstance(value,(int,float)):
                    writer.add_scalar('epoch/'+key,value,epoch)
            writer.flush()
            print(f'EPOCH {epoch}/100 COMPLETE '+str(row),flush=True)
            from .reporting import report
            report(cfg,stage='formal_training',trainer=trainer)
    finally:
        writer.close()
    return checkpoints/'best.pth'
