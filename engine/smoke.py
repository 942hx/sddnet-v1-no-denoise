"""Exactly 20 successful optimizer updates on real MILL train; never a formal experiment."""
from __future__ import annotations
import torch
import random
import time
import numpy as np
from PIL import Image,ImageDraw
from .training import Trainer, validate
from .profiling import profile_model
from utils.experiment import run_path
from utils.runtime import save_checkpoint
from utils.io import write_json, write_csv


def paired_figure(dataset,path):
    """Visualize real, synchronously transformed a/p/n low/GT pairs; preserve RNG."""
    state=random.getstate()
    try:
        random.seed(42)
        indices=dataset.sample_indices(0)
        random.seed(42)
        sample=dataset[0]
        size=dataset.patch_size
        canvas=Image.new('RGB',(3*size,2*size+48),'white')
        draw=ImageDraw.Draw(canvas)
        for col,(role,index) in enumerate(zip(('anchor','positive','negative'),indices)):
            draw.text((col*size+3,3),role+': '+dataset.records[index].filename,fill='black')
            for row,key in enumerate(('low','gt')):
                array=(sample[role+'_'+key].permute(1,2,0).numpy()*255).round().astype(np.uint8)
                canvas.paste(Image.fromarray(array),(col*size,24+row*(size+24)))
        canvas.save(path)
    finally:
        random.setstate(state)


def real_smoke(cfg):
    directory = run_path(cfg)/'smoke'
    if (directory/'smoke_results.json').exists() or (directory/'last.pth').exists():
        directory=directory/('attempt_'+time.strftime('%Y%m%dT%H%M%S',time.gmtime()))
    directory.mkdir(parents=True,exist_ok=True)
    write_json(run_path(cfg)/'latest_smoke.json',dict(relative_directory=directory.relative_to(run_path(cfg)).as_posix()))
    if not torch.cuda.is_available():
        result = dict(status='SKIP',reason='CUDA unavailable; real CUDA+AMP smoke not accepted',
                      arch_revision='v1_clean',optimizer_updates=0,completed_epoch=0)
        write_json(directory/'smoke_results.json',result)
        return result
    trainer = Trainer(cfg)
    losses,count = trainer.epoch(1,max_updates=20)
    if trainer.optimizer_updates!=20:
        raise RuntimeError(f'Smoke expected 20 updates; actual={trainer.optimizer_updates}')
    if not all(torch.isfinite(parameter).all().item() for parameter in trainer.model.parameters()):
        raise FloatingPointError('Non-finite model parameter after smoke optimizer updates')
    validation = validate(trainer.model,trainer.validation,trainer.device,max_images=6)
    diagnostic = trainer.diagnose()
    paired_figure(trainer.dataset,directory/'paired_samples.png')
    for name in ('structure_branch','degradation_branch','illumination_head'):
        norm = diagnostic[name+'_grad_norm_rec_only']
        if not torch.isfinite(torch.tensor(norm)) or norm<=0:
            raise RuntimeError(f'Reconstruction-only gradient path not accepted: {name} norm={norm}')
    state = trainer.checkpoint()
    state['run_kind'] = 'smoke'
    save_checkpoint(directory/'last.pth',state)
    write_csv(directory/'updates.csv',trainer.logs)
    write_csv(directory/'diagnostics.csv',[diagnostic])
    result = dict(status='PASS',arch_revision='v1_clean',optimizer_updates=trainer.optimizer_updates,
        attempted_updates=trainer.attempted_updates,AMP_overflows=trainer.overflows,
        microbatches=trainer.global_step,anchor_samples=count,completed_epoch=0,amp=trainer.use_amp,
        peak_train_allocated_bytes=trainer.peak_memory,training_seconds=trainer.elapsed_seconds,
        losses=losses,validation_probe=validation,validation_scope='first 6 validation images; not a formal result',
        diagnostic=diagnostic,checkpoint='smoke/last.pth',formal_training_started=False)
    write_json(directory/'smoke_results.json',result)
    print('REAL MILL SMOKE: '+str(result),flush=True)
    del trainer,state
    torch.cuda.empty_cache()
    profile_model(cfg,directory/'last.pth',output_path=directory/'profile.json',scope='20-update smoke; not a 100-epoch experiment')
    return result
