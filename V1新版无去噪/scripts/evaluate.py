"""Explicit A0 test evaluation; full datasets run only when requested."""
import argparse
from utils.experiment import experiment_config,assert_environment,DATASETS
from utils.paths import new_output
from engine.release import BEST,FINAL_METADATA,checked_checkpoint,seed_evaluation_directory


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',default='configs/a0.yaml')
    p.add_argument('--checkpoint',default=str(BEST))
    p.add_argument('--dataset',choices=[*DATASETS,'all'],required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--device',choices=['auto','cpu','cuda'])
    p.add_argument('--max-images',type=int,help='Explicitly limited smoke evaluation, never a complete-test result.')
    p.add_argument('--mill-root');p.add_argument('--lolv1-root');p.add_argument('--lolv2-root')
    args=p.parse_args(argv)
    if args.max_images is not None and args.max_images<1:p.error('--max-images must be positive.')
    cfg=experiment_config(args.config);assert_environment(cfg)
    destination=new_output(args.output)
    checked_checkpoint(cfg,args.checkpoint)
    cfg['run_dir']=str(destination)
    if args.device:cfg['device']=args.device
    for name in ['mill_root','lolv1_root','lolv2_root']:
        if getattr(args,name):cfg['data'][name]=getattr(args,name)
    selected=DATASETS if args.dataset=='all' else (args.dataset,)
    if not all((FINAL_METADATA/(d+'.csv')).is_file() for d in selected):
        p.error('Original frozen evaluation manifests are unavailable.')
    seed_evaluation_directory(cfg,destination)
    from engine.evaluation import evaluate_all
    evaluate_all(cfg,args.checkpoint,datasets=selected,max_images=args.max_images,metadata_dir=FINAL_METADATA)


if __name__=='__main__':main()
