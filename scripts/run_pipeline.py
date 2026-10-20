"""Single v1_clean entry point. Use --check-only/--smoke-test for this task's acceptance boundary."""
from __future__ import annotations
import argparse
import contextlib
import os
import subprocess
import sys
import traceback
from utils.experiment import (experiment_config,assert_environment,active_project_runs,RunLock,
                              save_resolved,run_path,validate_checkpoint,manifest_identity)
from utils.runtime import load_checkpoint,PROJECT_ROOT
from utils.io import write_json


class Tee:
    def __init__(self,*streams):
        self.streams=streams
    def write(self,text):
        for stream in self.streams:
            stream.write(text)
            stream.flush()
    def flush(self):
        for stream in self.streams:
            stream.flush()


def execute(args):
    cfg=experiment_config(args.config)
    assert_environment(cfg)
    active=active_project_runs()
    if active:
        raise RuntimeError(f'Another project training/evaluation run is active. No changes or duplicate run: {active}')
    directory=run_path(cfg)
    mode='check_only' if args.check_only else 'smoke' if args.smoke_test else 'evaluation' if args.eval_only else 'formal_training'
    with RunLock(directory,clear_stale=args.clear_stale_lock):
        save_resolved(cfg)
        log_name='train.log' if mode=='formal_training' else mode+'.log'
        with (directory/log_name).open('a',encoding='utf-8') as log, contextlib.redirect_stdout(Tee(sys.stdout,log)), contextlib.redirect_stderr(Tee(sys.stderr,log)):
            write_json(directory/'status.json',dict(state=mode,pid=os.getpid(),arch_revision='v1_clean'))
            try:
                from scripts.check_env import check_environment
                from scripts.prepare_data import prepare_data
                check_environment(cfg)
                prepare_data(cfg)
                if args.check_only:
                    # Exclude every optimizer-update test from the no-update CLI mode.
                    command=[sys.executable,'-m','pytest','-q','-m','not updates',
                             '--junitxml='+str(directory/'acceptance.xml')]
                    process=subprocess.run(command,cwd=PROJECT_ROOT,text=True,encoding='utf-8',errors='replace',
                                           stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
                    print(process.stdout,flush=True)
                    process.check_returncode()
                elif args.smoke_test:
                    from engine.smoke import real_smoke
                    real_smoke(cfg)
                elif args.eval_only:
                    from engine.evaluation import evaluate_all
                    from engine.profiling import profile_model
                    evaluate_all(cfg,args.checkpoint)
                    profile_model(cfg,args.checkpoint,scope='formal checkpoint evaluation')
                else:
                    if args.resume:
                        state=load_checkpoint(args.resume)
                        validate_checkpoint(state,cfg,manifest_identity(cfg))
                        if state.get('run_kind')!='formal' or state['sampler']['position']!=state['sampler']['size']:
                            raise ValueError('Formal resume requires a completed formal epoch, not a partial smoke checkpoint.')
                    from engine.training import formal_training
                    from engine.evaluation import evaluate_all
                    from engine.profiling import profile_model
                    checkpoint=formal_training(cfg,args.resume)
                    evaluate_all(cfg,checkpoint)
                    profile_model(cfg,checkpoint,scope='formal validation-best checkpoint')
                write_json(directory/'status.json',dict(state=mode+'_complete',pid=os.getpid(),arch_revision='v1_clean'))
                from engine.reporting import report
                report(cfg,stage=mode+'_complete')
            except BaseException as error:
                traceback.print_exc()
                write_json(directory/'status.json',dict(state='failed',mode=mode,reason=f'{type(error).__name__}: {error}'))
                from engine.reporting import report
                report(cfg,stage=mode+'_failed')
                raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default='configs/sddnet_v1.yaml')
    modes=parser.add_mutually_exclusive_group()
    modes.add_argument('--check-only',action='store_true',help='No optimizer creation/update; audit and no-update pytest cases')
    modes.add_argument('--smoke-test',action='store_true',help='Exactly 20 successful real-data optimizer updates, no formal training')
    modes.add_argument('--eval-only',action='store_true',help='No optimizer or training; checkpoint evaluation only')
    parser.add_argument('--resume')
    parser.add_argument('--checkpoint')
    parser.add_argument('--clear-stale-lock',action='store_true')
    args=parser.parse_args()
    if args.resume and (args.check_only or args.smoke_test or args.eval_only):
        parser.error('--resume is only valid for formal cumulative training')
    if args.eval_only != bool(args.checkpoint):
        parser.error('--eval-only requires --checkpoint, which is valid only in eval-only mode')
    execute(args)


if __name__=='__main__':
    main()
