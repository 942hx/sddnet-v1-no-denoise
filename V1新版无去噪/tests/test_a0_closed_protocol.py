"""Closed research protocols cannot silently enter the A0-only release."""
import copy
import inspect
import pytest
import torch
from models import SDDNetV1
from utils.experiment import experiment_config
from utils.runtime import load_config
from engine.training import Trainer


@pytest.mark.parametrize('field', ['generalization', 'stage6', 'experiment_protocol', 'lambda_identity'])
def test_closed_protocol_config_is_rejected_even_with_a0_budget(monkeypatch, field):
    cfg=copy.deepcopy(load_config('configs/a0.yaml'))
    cfg[field]={}
    monkeypatch.setattr('utils.experiment.load_config',lambda path: cfg)
    with pytest.raises(ValueError,match='closed experimental protocols'):
        experiment_config('not_read.yaml')


@pytest.mark.parametrize('kwargs', [dict(reconstruction_c_stopgrad=True),dict(scalar_condition_source='constant05')])
def test_model_has_no_sg_or_constant_execution_switch(kwargs):
    with pytest.raises(TypeError): SDDNetV1(**kwargs)


def test_one_original_trainer_without_extra_forward_or_sampler_dispatch():
    source=inspect.getsource(Trainer)
    assert 'backward_identity' not in source
    assert 'CIDNet' not in source and 'Stage6Triplets' not in source
    assert 'T_max=100' in source
    assert 'self.attempted_updates >= 10 and self.overflows / self.attempted_updates > 0.2' in source


def test_temporary_initialized_a0_reconstruction_reaches_condition_branches():
    from losses import SDDLoss
    torch.manual_seed(42)
    model=SDDNetV1().cpu().float()
    before={key:value.detach().clone() for key,value in model.state_dict().items()}
    raw=model(torch.rand(1,3,16,16),True)
    loss=SDDLoss().reconstruction(raw['enhanced'],torch.rand(1,3,16,16))[0]
    prefixes=['illumination_head','degradation_branch','bottleneck_affine.global_mlp',
              'skip_affine2.global_mlp','skip_affine1.global_mlp','skip_affine0.global_mlp']
    params=list(model.named_parameters())
    grads=torch.autograd.grad(loss,[p for _,p in params],allow_unused=True)
    for prefix in prefixes:
        selected=[g for (name,_),g in zip(params,grads) if name.startswith(prefix) and g is not None]
        assert selected and all(torch.isfinite(g).all() for g in selected)
        assert sum(g.abs().sum().item() for g in selected)>0
    assert all(torch.equal(value,model.state_dict()[key]) for key,value in before.items())
