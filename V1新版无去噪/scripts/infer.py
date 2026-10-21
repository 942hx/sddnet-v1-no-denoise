"""Enhance a single image or a folder using the strictly loaded A0 model."""
import argparse
from pathlib import Path
from utils.experiment import experiment_config,assert_environment
from utils.runtime import project_path
from utils.paths import new_output
from utils.io import write_json
from engine.release import BEST,load_model


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',default='configs/a0.yaml');p.add_argument('--checkpoint',default=str(BEST))
    p.add_argument('--input',required=True);p.add_argument('--output',required=True)
    p.add_argument('--device',choices=['auto','cpu','cuda'])
    args=p.parse_args(argv)
    cfg=experiment_config(args.config);assert_environment(cfg)
    destination=new_output(args.output);source=project_path(args.input)
    suffixes={'.png','.jpg','.jpeg','.bmp','.tif','.tiff'}
    paths=[source] if source.is_file() else sorted(x for x in source.iterdir() if x.is_file() and x.suffix.lower() in suffixes) if source.is_dir() else []
    if not paths or any(x.suffix.lower() not in suffixes for x in paths):p.error('No supported input images.')
    if len({x.stem.casefold() for x in paths})!=len(paths):p.error('Input basenames collide after PNG export.')
    model,state,device=load_model(cfg,args.checkpoint,args.device)
    destination.mkdir(parents=True,exist_ok=False)
    from engine.inference import infer_files
    rows=infer_files(model,device,paths,destination)
    write_json(destination/'inference.json',dict(variant='A0',checkpoint_epoch=state['completed_epoch'],images=rows,
        protocol='FP32 native RGB; raw forward, outer clamp, round-to-nearest PNG; no tiles/TTA/resize/GT adjustment'))


if __name__=='__main__':main()
