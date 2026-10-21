"""Frozen A0 identity and shared input/output checks for the thin CLIs."""
from pathlib import Path
import shutil
import torch
from models import SDDNetV1
from models.sddnet_v1 import model_options
from utils.experiment import (file_hash,read_json,validate_formal_checkpoint,manifest_identity)
from utils.runtime import PROJECT_ROOT,project_path,load_checkpoint,choose_device
from utils.paths import new_output,output_path

RELEASE=PROJECT_ROOT/'results/A0_best88'
BEST=RELEASE/'checkpoints/best.pth'
BEST_SHA256='a24a56157420d3868ab64ab75d4e91066084237ea9bfc082d4449a666c29b083'
TRAINING=RELEASE/'training/A0_seed42'
FINAL_METADATA=RELEASE/'evaluation/frozen_review/final_data/metadata'


def training_hashes():
    cfg=dict(run_dir=str(TRAINING))
    # Immutable evidence is read directly; runtime output guards intentionally
    # reject results/ as a write destination.
    audit=read_json(TRAINING/'data_audit.json')
    if not audit['passed']:raise ValueError('Frozen development audit failed.')
    for name,expected in audit['manifest_hashes'].items():
        if file_hash(TRAINING/'metadata'/f'{name}.csv')!=expected:
            raise ValueError('Frozen development manifest changed.')
    return audit['manifest_hashes']


def checked_checkpoint(cfg,path=BEST,for_resume=False):
    path=project_path(path)
    if path.resolve()==BEST.resolve() and file_hash(path)!=BEST_SHA256:
        raise ValueError('Frozen A0 best88 file SHA256 mismatch.')
    state=load_checkpoint(path)
    validate_formal_checkpoint(state,cfg,training_hashes(),for_resume=for_resume)
    if state.get('variant')!='A0' or state.get('seed')!=42:
        raise ValueError('Expected A0 seed42 checkpoint.')
    return state


def load_model(cfg,path=BEST,device=None):
    state=checked_checkpoint(cfg,path)
    device=choose_device(device or cfg['device'])
    model=SDDNetV1(**model_options(cfg)).float().to(device).eval()
    model.load_state_dict(state['model'],strict=True)
    return model,state,device


def seed_evaluation_directory(cfg,destination):
    """Copy only the tiny original development audit, not data or model files."""
    destination=output_path(destination)
    (destination/'metadata').mkdir(parents=True,exist_ok=False)
    source=TRAINING/'metadata/mill.csv'
    shutil.copyfile(source,destination/'metadata/mill.csv')
    shutil.copyfile(TRAINING/'data_audit.json',destination/'data_audit.json')
    if file_hash(destination/'metadata/mill.csv')!=file_hash(source):
        raise ValueError('Copied development manifest differs.')


def check_pair_files(root,row):
    """Authenticate every evaluated input/GT against the original audit."""
    for kind in ['input','gt']:
        path=root/row[kind+'_path']
        if not path.resolve().is_relative_to(root.resolve()):
            raise ValueError('Dataset manifest escapes its configured root.')
        if file_hash(path)!=row[kind+'_sha256']:
            raise ValueError('Dataset image differs from its original audit: '+str(path))
