"""Fixed-content validation interventions. Diagnostic only, never a training loss."""
from collections import defaultdict
from pathlib import Path
import math
import numpy as np
import torch
from torch.nn import functional as F
from models import ConditionalAffine
from losses import SDDLoss
from utils.io import write_csv,write_json
from utils.metrics import psnr_rgb
from losses.ssim import ssim


def derangement(count,seed=42):
    """Cyclic permutation of a seeded ordering: no fixed points, even with N=2."""
    if count<2: raise ValueError('Condition replacement needs at least two validation images.')
    order=np.random.default_rng(seed).permutation(count)
    result=np.empty(count,dtype=int);result[order]=np.roll(order,1)
    assert np.all(result!=np.arange(count))
    return result.tolist()


def center_pair(dataset,index,size):
    sample=dataset[index];low,gt=sample['low'],sample['gt']
    crop=min(size,*low.shape[-2:]);h,w=low.shape[-2:]
    top,left=(h-crop)//2,(w-crop)//2
    return low[:,top:top+crop,left:left+crop][None],gt[:,top:top+crop,left:left+crop][None],(top,left,crop,crop)


def validation_panel(dataset,contents=3):
    """All levels of the first fixed contents, never test/train donors."""
    if any(row.get('split','validation')!='validation' for row in dataset.rows):
        raise ValueError('Condition panel and donors must come only from validation.')
    scenes=sorted({r['content_id'] for r in dataset.rows})[:contents]
    indices=sorted([i for i,r in enumerate(dataset.rows) if r['content_id'] in scenes],
                   key=lambda i:(dataset.rows[i]['content_id'],float(dataset.rows[i]['level'])))
    if len(scenes)<2: raise ValueError('Both donor strata require >=2 validation contents.')
    return indices


def donor_indices(rows):
    same,other=[],[]
    for i,row in enumerate(rows):
        level=float(row['level'])
        matching=[j for j,r in enumerate(rows) if r['content_id']==row['content_id'] and float(r['level'])!=level]
        different=[j for j,r in enumerate(rows) if r['content_id']!=row['content_id'] and float(r['level'])==level
                   and not(row.get('group_id') and row['group_id']==r.get('group_id'))]
        if not matching or not different: raise ValueError('Panel has no legal donor for both validation strata.')
        # Predeclared farthest level; deterministic tie break, never optimized on output scores.
        same.append(max(matching,key=lambda j:(abs(float(rows[j]['level'])-level),-j)))
        other.append(different[0])
    assert all(i!=j for i,j in enumerate(same)) and all(i!=j for i,j in enumerate(other))
    return dict(same_content_different_level=same,different_content_same_level=other)


def affine_observer(model,destination,source):
    """Return removable hooks. Effective gamma=1+offset; std of offset is separate."""
    handles=[]
    for name,module in model.named_modules():
        if not isinstance(module,ConditionalAffine): continue
        def collect(mod,args,output,name=name):
            offset,beta,_,_=mod.affine_parameters(*args)
            row=dict(filename=source,layer=name,affine_mode=mod.affine_mode,
                     gain=mod.rho.sigmoid().item() if mod.rho is not None else 'NA')
            for key,value in (('gamma',1+offset),('delta_gamma',offset),('beta',beta)):
                v=value.float()
                row.update({key+'_mean':v.mean().item(),key+'_std':v.std(unbiased=False).item(),
                    key+'_p99':torch.quantile(v.flatten(),.99).item(),key+'_max':v.max().item(),
                    key+'_abs_p99':torch.quantile(v.abs().flatten(),.99).item(),key+'_abs_max':v.abs().max().item()})
            feature=args[0].float()
            row['modulation_delta_over_feature_RMS']=((output.float()-feature).square().mean().sqrt() /
                feature.square().mean().sqrt().clamp_min(1e-6)).item()
            destination.append(row)
        handles.append(module.register_forward_hook(collect))
    return handles


def reconstruction_gradient_probe(model,low,gt):
    """FP32 autograd.grad; does not populate .grad or mix auxiliary gradients."""
    previous=model.training;model.eval()
    try:
        with torch.inference_mode(False),torch.enable_grad(),torch.autocast(device_type=low.device.type,enabled=False):
            output=model(low.float(),True)
            rec=SDDLoss().reconstruction(output['enhanced'],gt.float())[0]
            named=[(n,p) for n,p in model.named_parameters() if n.startswith(
                ('structure_branch.','degradation_branch.','illumination_head.'))]
            grads=torch.autograd.grad(rec,[p for _,p in named],allow_unused=True)
            result={}
            for prefix in ('structure_branch','degradation_branch','illumination_head'):
                terms=[g.float().square().sum() for (n,_),g in zip(named,grads) if n.startswith(prefix+'.') and g is not None]
                result[prefix+'_grad_norm_rec_only']=torch.stack(terms).sum().sqrt().item() if terms else 'NA: no path'
            return result
    finally: model.train(previous)


def mean_quality(rows):
    result={}
    for name in ('PSNR_RGB','SSIM_RGB','raw_output_MAE','clamped_output_MAE',
                 'condition_quality_drop','condition_ssim_drop'):
        values=[r[name] for r in rows]
        result[name]=sum(values)/len(values) if values and all(isinstance(v,(int,float)) and math.isfinite(v) for v in values) else 'NA'
    return result


def summarize(rows):
    groups=defaultdict(list)
    for row in rows:
        base=(row['mode'],row['stratum'])
        for scope,scene,level in (('overall','ALL','ALL'),('level','ALL',row['level']),
                                  ('content',row['content_id'],'ALL'),('content_level',row['content_id'],row['level'])):
            groups[base+(scope,scene,level)].append(row)
    return [dict(mode=key[0],stratum=key[1],scope=key[2],content_id=key[3],level=key[4],count=len(values),
                 **mean_quality(values)) for key,values in groups.items()]


def run_condition_diagnostics(model,dataset,directory,size=128,contents=3,seed=42,provenance=None):
    """Cache ONLY detached donor Fd/c on CPU; hold one anchor's fixed encode.

    These are crop diagnostics, not full-image formal scores or causal proofs.
    No change is required for 'success'; harmful/wrong-condition improvements
    remain in CSV as measured. Gate c is replaced together with affine c.
    """
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    previous=model.training;model.eval();device=next(model.parameters()).device
    indices=validation_panel(dataset,contents);metadata=[dataset.rows[i] for i in indices]
    donors=donor_indices(metadata);donors['random_derangement']=derangement(len(indices),seed)
    conditions=[];embeddings=[];all_rows=[];affine_rows=[];grad_result={}
    try:
        with torch.inference_mode():
            for index in indices:
                low,_,_=center_pair(dataset,index,size)
                cache=model._encode(low.to(device))
                conditions.append((cache['Fd'].cpu(),cache['c_hat'].cpu()))
                embeddings.append(F.normalize(cache['Fs'].float().mean((2,3)),dim=1,eps=1e-6).cpu())
                del cache
            for pos,index in enumerate(indices):
                row=metadata[pos];low,gt,crop=center_pair(dataset,index,size)
                low,gt=low.to(device),gt.to(device);cache=model._encode(low)
                handles=affine_observer(model,affine_rows,row['filename'])
                try: normal,delta,gate=model._decode(cache)
                finally:
                    for h in handles:h.remove()
                torch.testing.assert_close(normal,model(low)['enhanced'],rtol=0,atol=0)
                normal_psnr=psnr_rgb(normal.clamp(0,1),gt);normal_ssim=ssim(normal.clamp(0,1),gt).item()
                modes=[('normal','none',None,{}),('neutral_affine','none',None,dict(affine_enabled=False)),
                       ('gate_only_off','none',None,dict(gate_enabled=False)),
                       ('all_condition_off','none',None,dict(affine_enabled=False,gate_enabled=False))]
                for stratum,permutation in donors.items():
                    donor=permutation[pos];fd,c=conditions[donor];fd,c=fd.to(device),c.to(device)
                    for mode,kwargs in (('fd_only',dict(fd=fd)),('c_only',dict(c_hat=c)),('joint',dict(fd=fd,c_hat=c))):
                        if stratum=='random_derangement' and mode!='joint':continue
                        modes.append((mode,stratum,donor,kwargs))
                for mode,stratum,donor,kwargs in modes:
                    changed,changed_delta,changed_gate=(normal,delta,gate) if mode=='normal' else model._decode(cache,**kwargs)
                    if not torch.isfinite(changed).all():raise FloatingPointError(f'Nonfinite condition diagnostic: {row["filename"]} {mode}')
                    clipped=changed.clamp(0,1)
                    score=psnr_rgb(clipped,gt);quality=ssim(clipped,gt).item()
                    result=dict(filename=row['filename'],split='validation',content_id=row['content_id'],
                        level=float(row['level']),group_id=row.get('group_id',''),mode=mode,stratum=stratum,
                        donor_filename=metadata[donor]['filename'] if donor is not None else '',
                        donor_content_id=metadata[donor]['content_id'] if donor is not None else '',
                        donor_level=metadata[donor]['level'] if donor is not None else '',
                        artificial_inconsistent_condition=mode in ('fd_only','c_only'),
                        PSNR_RGB=score,SSIM_RGB=quality,
                        raw_output_MAE=(changed-normal).abs().mean().item(),
                        clamped_output_MAE=(clipped-normal.clamp(0,1)).abs().mean().item(),
                        condition_quality_drop=normal_psnr-score,condition_ssim_drop=normal_ssim-quality,
                        gate_mean=changed_gate.mean().item(),delta_abs_mean=changed_delta.abs().mean().item(),
                        gated_delta_abs_mean=(changed_gate[:,:,None,None]*changed_delta).abs().mean().item(),
                        c_hat_condition_used=kwargs.get('c_hat',cache['c_hat']).item(),
                        c_hat=cache['c_hat'].item(),illumination_abs_error=abs(cache['c_hat'].item()-(float(row['level'])-1)/9),
                        raw_out_of_range_fraction=((changed<0)|(changed>1)).float().mean().item(),
                        brightness_signed_error=(clipped.mean()-gt.mean()).item(),
                        rgb_channel_mean_error_abs=(clipped.mean((2,3))-gt.mean((2,3))).abs().mean().item(),
                        saturation_fraction=((clipped<=0)|(clipped>=1)).float().mean().item(),
                        crop_top=crop[0],crop_left=crop[1],crop_height=crop[2],crop_width=crop[3])
                    all_rows.append(result)
                restored=model._decode(cache)[0]
                torch.testing.assert_close(restored,normal,rtol=0,atol=0)
                del cache,normal,changed,restored
        low,gt,_=center_pair(dataset,indices[0],size)
        grad_result=reconstruction_gradient_probe(model,low.to(device),gt.to(device))
        write_csv(directory/'condition_per_image.csv',all_rows)
        write_csv(directory/'condition_summary.csv',summarize(all_rows))
        write_csv(directory/'affine_statistics.csv',affine_rows)
        write_json(directory/'reconstruction_gradients.json',grad_result)
        write_json(directory/'fixed_validation_manifest.json',dict(seed=seed,crop='center',size=size,
            rows=[dict(dataset_index=i,filename=r['filename'],content_id=r['content_id'],level=r['level']) for i,r in zip(indices,metadata)],
            donor_maps=donors,all_donors_validation_only=True))
        result=dict(provenance=provenance or {},images=len(indices),rows=len(all_rows),scope='fixed validation crop diagnostics; not formal image metrics',
            formula=model.affine_mode,gate_enabled=model.residual_gate,restored_exactly=True,
            interpretations='Output sensitivity or triplet activity alone is not success or causal disentanglement.',
            reconstruction_gradients=grad_result)
        write_json(directory/'diagnostic_status.json',result)
        return result
    finally:model.train(previous)
