"""Train A0 on MILL train/validation only. Never starts final test evaluation."""
import argparse
import contextlib
import sys
from pathlib import Path
from utils.experiment import experiment_config,assert_environment,RunLock,save_resolved,manifest_identity
from utils.runtime import project_path,load_checkpoint
from utils.paths import output_path,new_output


def nonnegative_int(value):
    value=int(value)
    if value<0:raise argparse.ArgumentTypeError('Expected a nonnegative integer.')
    return value


def build_parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',default='configs/a0.yaml')
    p.add_argument('--output',required=True)
    p.add_argument('--resume')
    p.add_argument('--grad-norm-every',type=nonnegative_int)
    p.add_argument('--clear-stale-lock',action='store_true')
    return p


def main(argv=None):
    p=build_parser();args=p.parse_args(argv)
    cfg=experiment_config(args.config);assert_environment(cfg)
    directory=output_path(args.output)
    if args.resume:
        state_path=project_path(args.resume).resolve()
        if state_path!=directory/'checkpoints/last.pth':
            p.error('Resume must use this same run output/checkpoints/last.pth; best/smoke/frozen evidence cannot be resumed in place.')
        from utils.experiment import validate_formal_checkpoint,training_contract,read_json
        state=load_checkpoint(state_path)
        validate_formal_checkpoint(state,cfg,for_resume=True)
        if state.get('variant')!='A0' or not {'optimizer','scheduler','scaler','rng','loader_generator'}<=state.keys():
            p.error('Resume requires the complete A0 last training state.')
        if state['completed_epoch']>=100:p.error('This run already completed 100 epochs.')
    else:
        new_output(directory)
    cfg['run_dir']=str(directory)
    if args.grad_norm_every is not None:cfg['diagnostics']['grad_norm_every']=args.grad_norm_every
    from utils.runtime import choose_device
    if choose_device(cfg['device']).type!='cuda':p.error('Formal training requires CUDA.')
    from engine.training import formal_training,validate_best_checkpoint
    with RunLock(directory,clear_stale=args.clear_stale_lock):
        if args.resume:
            from utils.io import read_csv
            from utils.experiment import file_hash
            manifest_identity(cfg)
            root=project_path(cfg['data']['mill_root'])
            from engine.release import check_pair_files
            for row in read_csv(directory/'metadata/mill.csv'):check_pair_files(root,row)
        else:
            from scripts.prepare_data import prepare_data
            prepare_data(cfg,scope='development')
        save_resolved(cfg)
        class Tee:
            def __init__(self,*streams):self.streams=streams
            def write(self,text):
                for stream in self.streams:stream.write(text);stream.flush()
            def flush(self):
                for stream in self.streams:stream.flush()
        with (directory/'train.log').open('a' if args.resume else 'x',encoding='utf-8') as log:
            with contextlib.redirect_stdout(Tee(sys.stdout,log)),contextlib.redirect_stderr(Tee(sys.stderr,log)):
                checkpoint=formal_training(cfg,args.resume)
                validate_best_checkpoint(cfg,checkpoint)


if __name__=='__main__':main()
