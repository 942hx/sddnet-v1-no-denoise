"""Small validation probes: sensitivity, feature statistics and reconstruction-only gradients."""
from __future__ import annotations
from contextlib import contextmanager
import torch
from torch.nn import functional as F
from models import ConditionalAffine
from losses import SDDLoss
from utils.metrics import psnr_rgb


def branch_gradient_norms(model):
    """Unscaled pre-clip norms, with one batched device-to-host transfer.

    The caller samples only successful optimizer updates. A branch with no
    gradients is missing (NaN), distinct from a measured zero gradient norm.
    """
    names = ('structure_branch', 'degradation_branch', 'illumination_head', 'bottleneck_affine',
             'skip_affine2', 'skip_affine1', 'skip_affine0', 'head')
    norms = []
    device = next(model.parameters()).device
    for name in names:
        squares = [p.grad.detach().float().square().sum() for p in getattr(model, name).parameters() if p.grad is not None]
        norms.append(torch.stack(squares).sum().sqrt() if squares else
                     torch.full((),float('nan'),device=device,dtype=torch.float32))
    values = torch.stack(norms).detach().cpu().tolist()
    return dict(zip((name+'_grad_norm_total' for name in names),values))


@contextmanager
def condition_intervention(model, mode, permutation=None):
    """Temporary hooks only; weights and public forward signature remain unchanged."""
    if mode not in ('identity', 'shuffle_c', 'shuffle_fd'):
        raise ValueError('Unknown intervention')
    if mode=='shuffle_c' and getattr(model,'residual_gate',False):
        raise ValueError('Legacy affine hooks cannot replace the output gate condition; use fixed-content run_condition_diagnostics.')
    handles = []
    original_training = model.training
    model.eval()
    try:
        for module in model.modules():
            if not isinstance(module, ConditionalAffine):
                continue
            if mode == 'identity':
                handles.append(module.register_forward_hook(lambda mod, args, output: args[0]))
            else:
                def prehook(mod, args):
                    feature, fd, c = args
                    if feature.shape[0] < 2 or permutation is None:
                        raise ValueError('Shuffle needs >=2 samples and an explicit non-identity permutation.')
                    if len(permutation)!=feature.shape[0] or (permutation == torch.arange(len(permutation), device=permutation.device)).any():
                        raise ValueError('Every condition must change: fixed points are not allowed.')
                    return (feature, fd, c[permutation]) if mode == 'shuffle_c' else (feature, fd[permutation], c)
                handles.append(module.register_forward_pre_hook(prehook))
        yield
    finally:
        for handle in handles:
            handle.remove()
        model.train(original_training)


def fixed_probe(dataset, size=128):
    """Two aligned levels of one validation content plus another content at the anchor level."""
    rows = dataset.rows
    first = rows[0]['content_id']
    same = sorted((i for i, row in enumerate(rows) if row['content_id'] == first), key=lambda i: float(rows[i]['level']))
    anchor, positive = same[0], same[-1]
    if rows[anchor]['level'] == rows[positive]['level']:
        raise ValueError('Probe needs two distinct validation levels for one content.')
    negative = next(i for i, row in enumerate(rows) if row['content_id'] != first and row['level'] == rows[anchor]['level'])
    lows, targets, levels = [], [], []
    for index in (anchor, positive, negative):
        sample = dataset[index]
        low, gt = sample['low'], sample['gt']
        h, w = low.shape[-2:]
        crop = min(size, h, w)
        top, left = (h-crop)//2, (w-crop)//2
        lows.append(low[:, top:top+crop, left:left+crop])
        targets.append(gt[:, top:top+crop, left:left+crop])
        levels.append([(float(rows[index]['level'])-1)/9])
    return torch.stack(lows), torch.stack(targets), torch.tensor(levels, dtype=torch.float32)


def diagnostics(model, low, gt, labels, criterion, training_median=0.5, total_gradient_norms=None):
    """Probe metrics are diagnostic-only, never merged into full validation/test scores."""
    original_training = model.training
    model.eval()
    result = dict(scope='fixed validation center crop; not formal full-image metrics', probe_images=len(low))
    handles = []
    try:
        for name, module in model.named_modules():
            if isinstance(module, ConditionalAffine):
                def stats_hook(mod, args, output, name=name):
                    gamma, beta, gr, br = mod.affine_parameters(*args)
                    result[name+'_a'] = mod.rho.sigmoid().item()
                    for key, value in (('gamma', gamma), ('beta', beta)):
                        result[name+'_'+key+'_mean'] = value.float().mean().item()
                        result[name+'_'+key+'_abs_mean'] = value.float().abs().mean().item()
                    result[name+'_tanh_saturation_fraction'] = (torch.cat([gr.tanh().flatten(), br.tanh().flatten()]).abs().gt(.99).float().mean().item()
                        if mod.affine_mode=='bounded' else 'NA: no tanh')
                handles.append(module.register_forward_hook(stats_hook))
        with torch.no_grad():
            output = model(low, return_latents=True)
            for handle in handles:
                handle.remove()
            handles.clear()
            for key in ('Fs', 'Fd'):
                feature = output[key].float()
                result[key+'_mean'] = feature.mean().item()
                result[key+'_std'] = feature.std(unbiased=False).item()
                result[key+'_norm'] = feature.norm().item()
            embeddings = F.normalize(output['Fs'].float().mean((2,3)), dim=1, eps=1e-6)
            result['embedding_variance_across_contents'] = embeddings[[0,2]].var(dim=0, unbiased=False).mean().item()
            result['same_content_cross_level_distance'] = (embeddings[0]-embeddings[1]).norm().item()
            result['different_content_same_level_distance'] = (embeddings[0]-embeddings[2]).norm().item()
            result['c_hat_std'] = output['c_hat'].float().std(unbiased=False).item()
            result['illumination_MAE'] = (output['c_hat'].float()-labels).abs().mean().item()
            result['median_constant_MAE'] = (labels-training_median).abs().mean().item()
            result['y_raw_out_of_range_fraction'] = ((output['enhanced']<0)|(output['enhanced']>1)).float().mean().item()
            normal = output['enhanced'].detach().clone()
            result['probe_PSNR_normal'] = sum(psnr_rgb(a[None].clamp(0,1), b[None]) for a,b in zip(normal,gt))/len(gt)
            permutation = torch.arange(len(low), device=low.device).roll(1)
            cache=model._encode(low)
            for mode in ('identity', 'shuffle_c', 'shuffle_fd'):
                kwargs={'affine_enabled':False} if mode=='identity' else (
                    {'c_hat':cache['c_hat'][permutation]} if mode=='shuffle_c' else {'fd':cache['Fd'][permutation]})
                changed=model._decode(cache,**kwargs)[0]
                score = sum(psnr_rgb(a[None].clamp(0,1),b[None]) for a,b in zip(changed,gt))/len(gt)
                result[mode+'_raw_output_MAE'] = (changed-normal).abs().mean().item()
                result[mode+'_PSNR_delta'] = score-result['probe_PSNR_normal']
            restored = model(low)['enhanced']
            torch.testing.assert_close(restored, normal, rtol=0, atol=0)
            result['interventions_restored_exactly'] = True
            del output, normal, restored, changed
        # Independent FP32 graph: auxiliary losses cannot mask a missing reconstruction path.
        with torch.enable_grad(), torch.autocast(device_type=low.device.type, enabled=False):
            output = model(low.float(), return_latents=True)
            rec = criterion.reconstruction(output['enhanced'], gt.float())[0]
            selected = [(name, p) for name,p in model.named_parameters()
                        if name.startswith(('structure_branch.', 'degradation_branch.', 'illumination_head.'))]
            gradients = torch.autograd.grad(rec, [p for _,p in selected], allow_unused=True)
            for prefix in ('structure_branch', 'degradation_branch', 'illumination_head'):
                values = [grad.float().square().sum() for (name,_),grad in zip(selected, gradients)
                          if name.startswith(prefix+'.') and grad is not None]
                result[prefix+'_grad_norm_rec_only'] = torch.stack(values).sum().sqrt().item() if values else 0.0
        if total_gradient_norms:
            result.update(total_gradient_norms)
        result['interpretation'] = 'Sensitivity/dependency checks only; not a causal disentanglement proof.'
        return result
    finally:
        for handle in handles:
            handle.remove()
        model.train(original_training)
