"""v1_clean acceptance. Synthetic fixtures validate code and are never experiment results."""
from __future__ import annotations
import copy
import math
from pathlib import Path
import numpy as np
import pytest
import torch
from PIL import Image
from torch.nn import functional as F
from skimage.metrics import structural_similarity
from models import SDDNetV1,ConditionalAffine
from models.sddnet_v1 import brightness_prior
from losses import SDDLoss,ssim
from losses.sdd_loss import consistency_loss
from datasets import MILLTripletDataset
from datasets.sampling import AnchorSampler
from engine.training import optimizer_for,accumulation_samples
from engine.diagnostics import diagnostics,condition_intervention
from utils.runtime import capture_rng,restore_rng,save_checkpoint,load_checkpoint,make_scaler,PROJECT_ROOT
from utils.experiment import experiment_config,validate_checkpoint,training_contract,RunLock
from utils.metrics import psnr_rgb,load_lpips

torch.set_num_threads(2)


@pytest.fixture
def model():
    torch.manual_seed(42)
    return SDDNetV1()


@pytest.mark.parametrize('shape',[(1,3,256,256),(1,3,255,257),(1,3,400,600),(1,3,1,1),(1,3,2,3)])
def test_shapes_return_contract_prior_and_affine(model,shape):
    x=torch.rand(shape)
    h,w=shape[-2:]
    with torch.no_grad():
        full=model(x,return_latents=True)
        lean=model(x)
    assert set(full)=={'enhanced','Fs','Fd','c_hat'} and set(lean)=={'enhanced'}
    torch.testing.assert_close(full['enhanced'],lean['enhanced'],rtol=0,atol=0)
    assert full['enhanced'].shape==shape
    assert full['Fs'].shape==(shape[0],128,math.ceil(h/8),math.ceil(w/8))
    assert full['Fd'].shape==full['Fs'].shape and full['c_hat'].shape==(shape[0],1)
    assert brightness_prior(x).shape==(shape[0],2,h,w)
    assert model.degradation_branch[0].in_channels==258
    assert model.fusion[0].in_channels==128
    assert all(not isinstance(m,(torch.nn.BatchNorm2d,torch.nn.InstanceNorm2d,torch.nn.LayerNorm)) for m in model.modules())


@pytest.mark.parametrize('x,error',[(torch.rand(0,3,16,16),ValueError),(torch.rand(1,4,16,16),ValueError),
                                  (torch.ones(1,3,16,16,dtype=torch.uint8),TypeError),(torch.rand(3,16,16),ValueError)])
def test_invalid_input_messages(model,x,error):
    with pytest.raises(error,match='RGB|non-empty|channels'):
        model(x)


def test_prior_is_computed_before_padding(model):
    x=torch.rand(1,3,15,17)
    expected=torch.cat([F.avg_pool2d(x.amax(1,True),5,1,2,count_include_pad=False),
                        F.avg_pool2d(x.mean(1,True),5,1,2,count_include_pad=False)],1)
    torch.testing.assert_close(brightness_prior(x),expected)
    handles=[]
    seen={}
    for name in ('bottleneck_affine','skip_affine2','skip_affine1','skip_affine0'):
        def hook(module,args,out,name=name):
            gamma,beta,_,_=module.affine_parameters(*args)
            assert gamma.shape==beta.shape==args[0].shape
            seen[name]=module.rho.sigmoid().item()
        handles.append(getattr(model,name).register_forward_hook(hook))
    with torch.no_grad(): model(x)
    for handle in handles: handle.remove()
    assert len(seen)==4 and all(abs(a-.1)<1e-6 for a in seen.values())
    assert len({id(getattr(model,name).rho) for name in seen})==4


def fake_outputs(fs=None):
    return dict(enhanced=torch.rand(3,3,16,16,requires_grad=True),
                Fs=fs if fs is not None else torch.rand(3,128,2,2,requires_grad=True),
                Fd=torch.rand(3,128,2,2),c_hat=torch.rand(3,1,requires_grad=True))


def test_exact_label_shape_triplet_formula_and_warmup():
    criterion=SDDLoss()
    fs=torch.zeros(3,128,2,2)
    fs[0,0]=1; fs[1,1]=1; fs[2,0]=1
    fs.requires_grad_()
    outputs=fake_outputs(fs)
    target=torch.rand(3,3,16,16)
    with pytest.raises(ValueError,match=r'\[N,1\]'):
        criterion(outputs,target,torch.rand(3))
    losses=criterion(outputs,target,torch.rand(3,1),progress=2.5)
    embedding=F.normalize(fs.mean((2,3)),dim=1,eps=1e-6)
    reference=F.triplet_margin_loss(*embedding.chunk(3),margin=.2,p=2,eps=1e-6,swap=False)
    torch.testing.assert_close(losses['scene'],reference)
    assert losses['scene']>1.5  # Deliberately active unsquared-distance triplet.
    assert abs(losses['lambda_cons'].item()-.01)<1e-7
    assert criterion.consistency_weight(0)==0 and criterion.consistency_weight(5)==.02
    expected=losses['rec']+.1*losses['scene']+.05*losses['illumination']+.01*losses['cons']
    torch.testing.assert_close(losses['total'],expected)
    losses['scene'].backward()
    assert torch.isfinite(fs.grad).all() and fs.grad.abs().sum()>0


def test_consistency_cosine_and_unclamped_l1_gradient():
    feature=torch.rand(1,8,4,4,requires_grad=True)
    assert consistency_loss(feature,feature).item()<1e-6
    changed=feature.detach().clone()+.2*torch.randn_like(feature)
    consistency_loss(feature,changed).backward()
    assert torch.isfinite(feature.grad).all()
    raw=torch.full((1,3,16,16),2.0,requires_grad=True)
    rec,l1,_=SDDLoss().reconstruction(raw,torch.zeros_like(raw))
    rec.backward()
    assert l1.item()==2 and torch.all(raw.grad>0)  # clamp-only would have zero restoring gradient.


def test_reconstruction_paths_and_hook_restoration(model):
    low=torch.rand(3,3,32,40)
    gt=torch.rand_like(low)
    labels=torch.tensor([[0.],[1.],[0.]],dtype=torch.float32)
    model.train()
    original={k:v.detach().clone() for k,v in model.state_dict().items()}
    result=diagnostics(model,low,gt,labels,SDDLoss())
    assert model.training and result['interventions_restored_exactly']
    for prefix in ('structure_branch','degradation_branch','illumination_head'):
        assert result[prefix+'_grad_norm_rec_only']>0
    assert result['embedding_variance_across_contents']>=0
    for key,value in original.items(): torch.testing.assert_close(value,model.state_dict()[key],rtol=0,atol=0)
    with pytest.raises(RuntimeError):
        with condition_intervention(model,'identity'):
            raise RuntimeError('test restoration on exception')
    assert all(not module._forward_hooks and not module._forward_pre_hooks for module in model.modules())


def test_initialization_and_decay_groups(model):
    cfg=experiment_config('configs/sddnet_v1.yaml')
    for layer in (model.head,model.illumination_head[-2],model.bottleneck_affine.local):
        assert .0005<layer.weight.std().item()<.0015
        assert layer.bias.count_nonzero()==0
    assert all(torch.isfinite(p).all() for p in model.parameters())
    # Inspect groups without constructing an optimizer in --check-only.
    no_decay={name for name,p in model.named_parameters() if p.ndim<=1 or name.endswith('.bias')}
    assert all(name in no_decay for name,_ in model.named_parameters() if name.endswith('.rho'))


@pytest.mark.updates
def test_weight_resume_scheduler_scaler_rng_and_legacy_rejection(model,tmp_path):
    cfg=experiment_config('configs/sddnet_v1.yaml')
    optimizer=optimizer_for(model,cfg)
    no_decay_ids={id(p) for group in optimizer.param_groups if group['weight_decay']==0 for p in group['params']}
    assert all(id(p) in no_decay_ids for name,p in model.named_parameters() if p.ndim<=1)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,100,eta_min=1e-6)
    scaler=make_scaler(False)
    x=torch.rand(3,3,32,40); target=torch.rand_like(x); labels=torch.rand(3,1)
    criterion=SDDLoss()
    sampler=AnchorSampler(5);sampler.start_epoch(1);next(iter(sampler))
    def step():
        optimizer.zero_grad(set_to_none=True)
        criterion(model(x,True),target,labels,progress=1)['total'].backward()
        optimizer.step()
    step();scheduler.step()
    state=dict(model=model.state_dict(),optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),scaler=scaler.state_dict(),
               rng=capture_rng(),sampler=sampler.state_dict(),completed_epoch=1,global_step=1,
               arch_revision='v1_clean',contract=training_contract(cfg),manifest_hashes={'fixture':'unit'})
    path=tmp_path/'resume.pth';save_checkpoint(path,state)
    step();expected={k:v.detach().clone() for k,v in model.state_dict().items()}
    loaded=load_checkpoint(path);validate_checkpoint(loaded,cfg,{'fixture':'unit'})
    model.load_state_dict(loaded['model'],strict=True);optimizer.load_state_dict(loaded['optimizer'])
    scheduler.load_state_dict(loaded['scheduler']);scaler.load_state_dict(loaded['scaler']);restore_rng(loaded['rng'])
    sampler.load_state_dict(loaded['sampler']);assert sampler.position==1
    step()
    for key,value in expected.items():torch.testing.assert_close(value,model.state_dict()[key],rtol=0,atol=0)
    assert loaded['completed_epoch']==1 and scheduler.last_epoch==1
    with pytest.raises(ValueError,match='Legacy resume rejected'):
        validate_checkpoint(dict(loaded,arch_revision='old_v1'),cfg)


@pytest.mark.updates
def test_real_cuda_amp_accumulation_step(model):
    if not torch.cuda.is_available():pytest.skip('CUDA unavailable: CUDA/AMP is not accepted')
    model=model.cuda();cfg=experiment_config('configs/sddnet_v1.yaml')
    optimizer=optimizer_for(model,cfg);scaler=make_scaler(True);criterion=SDDLoss().cuda()
    optimizer.zero_grad(set_to_none=True)
    for _ in range(2):
        with torch.amp.autocast('cuda'):
            output=model(torch.rand(3,3,256,256,device='cuda'),True)
            loss=criterion(output,torch.rand(3,3,256,256,device='cuda'),torch.rand(3,1,device='cuda'))['total']
        assert torch.isfinite(loss)
        scaler.scale(loss/2).backward()
    scaler.unscale_(optimizer)
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
    assert torch.isfinite(norm)
    scaler.step(optimizer);scaler.update();optimizer.zero_grad(set_to_none=True)
    assert all(torch.isfinite(parameter).all() for parameter in model.parameters())
    restored_scaler=make_scaler(True)
    restored_scaler.load_state_dict(scaler.state_dict())
    assert restored_scaler.state_dict()==scaler.state_dict()


def test_sampler_and_accumulation_tail():
    sampler=AnchorSampler(5);sampler.start_epoch(1)
    first=next(iter(sampler));state=sampler.state_dict();rest=list(sampler)
    clone=AnchorSampler(5);clone.load_state_dict(state)
    assert list(clone)==rest and sorted([first]+rest)==list(range(5))
    assert [accumulation_samples(9,2,2,i) for i in range(5)]==[4,4,4,4,1]


def test_metadata_triplet_alignment_and_same_level_negative(tmp_path):
    for kind in ('input','gt'):(tmp_path/'train'/kind).mkdir(parents=True)
    pixels=np.arange(40*48*3).reshape(40,48,3).astype(np.uint8)
    for offset,content in enumerate(('Scene1_a','Scene1_b','Scene2')):
        image=Image.fromarray((pixels+offset).astype(np.uint8))
        image.save(tmp_path/'train'/'gt'/(content+'.png'))
        for level in (1,10):image.save(tmp_path/'train'/'input'/f'{content}-{level}.png')
    dataset=MILLTripletDataset(tmp_path,patch_size=24)
    for i in range(len(dataset)):
        a,p,n=[dataset.records[k] for k in dataset.sample_indices(i)]
        assert a.content_id==p.content_id and a.level!=p.level and a.content_id!=n.content_id and a.level==n.level
        sample=dataset[i]
        torch.testing.assert_close(sample['anchor_low'],sample['positive_low'])
        for role in ('anchor','positive','negative'):
            torch.testing.assert_close(sample[role+'_low'],sample[role+'_gt'])


def test_metrics_properties_and_reference():
    a=torch.rand(1,3,32,40);b=.8*a+.2*torch.rand_like(a)
    assert ssim(a,a).item()==1 and math.isinf(psnr_rgb(a,a))
    arrays=[value[0].permute(1,2,0).numpy() for value in (a,b)]
    reference=structural_similarity(*arrays,data_range=1,channel_axis=-1,gaussian_weights=True,sigma=1.5,
                                    use_sample_covariance=False,win_size=11,K1=.01,K2=.03)
    assert abs(ssim(a,b).item()-reference)<2e-5
    with pytest.raises(ValueError):ssim(a[:,:,:8,:8],a[:,:,:8,:8])


def test_pretrained_lpips_identical_image():
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    metric,status=load_lpips(device)
    if metric is None:pytest.skip('LPIPS unavailable: '+status['reason'])
    with torch.inference_mode():
        x=torch.rand(1,3,64,64,device=device)*2-1
        assert abs(metric(x,x).item())<1e-6


def test_run_lock(tmp_path):
    with RunLock(tmp_path):
        with pytest.raises(RuntimeError):
            with RunLock(tmp_path):pass
