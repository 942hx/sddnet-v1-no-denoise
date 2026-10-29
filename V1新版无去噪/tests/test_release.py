"""A0 release paths, frozen-file protection, CLI routing and strict identity."""
import copy
import json
import sys
from pathlib import Path
import pytest
import torch
from models import SDDNetV1,ConditionalAffine
from utils.runtime import PROJECT_ROOT
from utils.paths import output_path,new_output
from utils.experiment import experiment_config,training_contract,object_hash
from engine.release import BEST,BEST_SHA256,checked_checkpoint,training_hashes


def test_release_checkpoint_is_exact_frozen_best88():
    cfg=experiment_config('configs/a0.yaml');state=checked_checkpoint(cfg)
    assert state['variant']=='A0' and state['completed_epoch']==state['best_epoch']==88
    assert state['seed']==42 and state['config_hash']==object_hash(training_contract(cfg))
    assert len(state['model'])==148
    model=SDDNetV1();result=model.load_state_dict(state['model'],strict=True)
    assert not result.missing_keys and not result.unexpected_keys
    assert sum(p.numel() for p in model.parameters())==12270696


@pytest.mark.parametrize('destination',['..','../sibling/output','data/x','cache/x','results/A0_best88/x','.git/index'])
def test_protected_output_rejected_without_writes(destination):
    with pytest.raises(ValueError):output_path(destination)


@pytest.mark.parametrize('destination',[
    'DATA/x','Data/x','CACHE/x','Results/A0_best88/x','.GIT/index',
    '.Agents/x','.CODEX/x','.AWS/x','data./x','results /x',
    'outputs/.. /outside','outputs/file:stream'])
def test_windows_case_and_aliases_cannot_bypass_protected_output(destination):
    with pytest.raises(ValueError):output_path(destination)


def test_normal_chinese_and_internal_spaces_output_still_valid():
    path=output_path('outputs/中文 空格/增强')
    assert path.relative_to(PROJECT_ROOT).parts==('outputs','中文 空格','增强')


def test_existing_output_rejected_and_evidence_unchanged(tmp_path):
    path=tmp_path/'existing';path.mkdir();evidence=path/'metrics.csv';evidence.write_bytes(b'original\r\n')
    with pytest.raises(FileExistsError):new_output(path)
    assert evidence.read_bytes()==b'original\r\n'


def test_only_a0_architecture_options_are_supported():
    for kwargs in [dict(affine_mode='standard'),dict(residual_gate=True),dict(local_init_std=.0001)]:
        with pytest.raises(ValueError):SDDNetV1(**kwargs)
    with pytest.raises(ValueError):ConditionalAffine(8,1,affine_mode='standard')


def test_a0_bounded_affine_formula_and_rho():
    module=ConditionalAffine(8,1);feature=torch.rand(1,8,5,7);fd=torch.rand(1,128,2,3);c=torch.rand(1,1)
    offset,beta,raw_offset,raw_beta=module.affine_parameters(feature,fd,c)
    torch.testing.assert_close(offset,module.rho.sigmoid()*raw_offset.tanh(),rtol=0,atol=0)
    torch.testing.assert_close(beta,module.rho.sigmoid()*raw_beta.tanh(),rtol=0,atol=0)
    torch.testing.assert_close(module(feature,fd,c),feature*(1+offset)+beta,rtol=0,atol=0)


@pytest.mark.parametrize('module',['train','evaluate','infer','prepare_data'])
def test_cli_cannot_write_frozen_results(module):
    from importlib import import_module
    entry=import_module('scripts.'+module)
    argv=['--output','results/A0_best88/overwrite']
    if module=='evaluate':argv+=['--dataset','mill']
    if module=='infer':argv+=['--input','missing.png']
    with pytest.raises(ValueError):entry.main(argv)


def test_train_refuses_best_or_frozen_last_as_same_run_resume(tmp_path):
    from scripts.train import main
    with pytest.raises(SystemExit):main(['--output',str(tmp_path/'run'),'--resume',str(BEST)])
    assert not (tmp_path/'run').exists()


def test_evaluate_is_explicit_and_limits_are_not_silent():
    from scripts.evaluate import main
    with pytest.raises(SystemExit):main(['--output','outputs/not-created-by-test'])
    with pytest.raises(SystemExit):main(['--dataset','mill','--output','outputs/not-created-by-test','--max-images','0'])
    assert not (PROJECT_ROOT/'outputs/not-created-by-test').exists()


def test_cli_checkpoint_rejection_precedes_output_creation(monkeypatch,tmp_path):
    from scripts import evaluate
    def reject(*a,**k):raise ValueError('unit fixture: incompatible checkpoint')
    monkeypatch.setattr(evaluate,'checked_checkpoint',reject)
    with pytest.raises(ValueError,match='incompatible'):
        evaluate.main(['--dataset','mill','--output',str(tmp_path/'new')])
    assert not (tmp_path/'new').exists()


def test_normal_inference_does_not_accept_true_levels():
    model=SDDNetV1()
    with pytest.raises(TypeError):model(torch.rand(1,3,12,12),c_gt=torch.tensor([[1.]]))


def test_fixed_triplets_reject_other_layout_and_miner():
    from losses.mining import select_triplets
    x=torch.rand(3,4);info=dict(layout='triplet',content_ids=['a','a','b'],raw_levels=[1,10,1])
    for mode in ['random','semihard']:
        with pytest.raises(ValueError):select_triplets(x,info,mode,.2)
    with pytest.raises(ValueError):select_triplets(x,dict(info,layout='pk'),'fixed',.2)


def test_junction_output_rejected(tmp_path):
    """Use the actual Windows directory-junction API through its native shell."""
    import os,subprocess
    if os.name!='nt':pytest.skip('Windows junction test')
    target=tmp_path/'target';target.mkdir();link=tmp_path/'link'
    # Native cmd may emit UTF-8 or an OEM code page. Keep opaque command output
    # as bytes: the asserted behavior is the real junction, not its message.
    result=subprocess.run(['cmd','/c','mklink','/J',str(link),str(target)],capture_output=True)
    assert result.returncode==0,result.stderr
    try:
        with pytest.raises(ValueError):output_path(link/'output')
        assert not (target/'output').exists()
    finally:os.rmdir(link)
