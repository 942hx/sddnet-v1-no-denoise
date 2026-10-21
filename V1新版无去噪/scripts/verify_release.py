"""Read-only A0 release checks and one CPU small-tensor forward, no training."""
import argparse,json
import torch
from utils.experiment import experiment_config,file_hash,assert_environment
from engine.release import BEST,BEST_SHA256,load_model,RELEASE
from utils.runtime import project_path


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',default='configs/a0.yaml');p.add_argument('--checkpoint',default=str(BEST))
    args=p.parse_args(argv);cfg=experiment_config(args.config);assert_environment(cfg)
    model,state,device=load_model(cfg,args.checkpoint,device='cpu');torch.set_num_threads(2)
    with torch.inference_mode():
        out=model(torch.linspace(0,1,3*17*25).reshape(1,3,17,25),return_latents=True)
    if not all(torch.isfinite(v).all() for v in out.values()):raise FloatingPointError('Non-finite forward.')
    if state['completed_epoch']!=88 or file_hash(project_path(args.checkpoint))!=BEST_SHA256:
        raise ValueError('verify_release requires the exact frozen best88 file.')
    print(json.dumps(dict(status='PASS',variant='A0',epoch=88,sha256=BEST_SHA256,strict=True,
        parameters=sum(p.numel() for p in model.parameters()),raw_shape=list(out['enhanced'].shape),
        precision='CPU FP32',training_started=False),ensure_ascii=False,indent=2))


if __name__=='__main__':main()
