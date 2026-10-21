"""Shared FP32 whole-image inference; quantize only after the raw prediction."""
from pathlib import Path
import torch
from datasets.common import load_rgb,to_tensor
from .evaluation import png


@torch.inference_mode()
def infer_files(model,device,inputs,destination):
    rows=[]
    for path in inputs:
        low=to_tensor(load_rgb(path)).unsqueeze(0).to(device)
        raw=model(low)['enhanced']
        if raw.shape!=low.shape or not torch.isfinite(raw).all():
            raise FloatingPointError('Invalid raw prediction for '+str(path))
        target=destination/(path.stem+'.png')
        png(raw.clamp(0,1)).save(target)
        rows.append(dict(input=str(path),output=str(target),shape=list(raw.shape)))
    return rows
