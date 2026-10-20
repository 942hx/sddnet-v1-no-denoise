"""Separate acceptance/smoke evidence from formal experiment results; never invent missing numbers."""
from __future__ import annotations
import json
import xml.etree.ElementTree as ET
from utils.experiment import run_path,read_json
from utils.io import read_csv
from utils.runtime import load_checkpoint


def report(cfg,stage='acceptance',trainer=None):
    directory = run_path(cfg)
    smoke_dir=directory/'smoke'
    if (directory/'latest_smoke.json').exists():
        smoke_dir=directory/read_json(directory/'latest_smoke.json')['relative_directory']
    smoke_path = smoke_dir/'smoke_results.json'
    smoke = read_json(smoke_path) if smoke_path.exists() else {}
    profile_path = smoke_dir/'profile.json'
    if (directory/'profile.json').exists():
        profile_path = directory/'profile.json'
    profile = read_json(profile_path) if profile_path.exists() else {}
    history = read_csv(directory/'history.csv')
    summary = read_csv(directory/'metrics'/'summary.csv')
    diagnostics = read_csv(directory/'diagnostics.csv')
    completed = trainer.completed_epoch if trainer else int(history[-1]['completed_epoch']) if history else 0
    environment = read_json(directory/'environment.json') if (directory/'environment.json').exists() else {}
    tests = []
    junit = directory/'acceptance.xml'
    if (directory/'acceptance_all.xml').exists():
        junit = directory/'acceptance_all.xml'
    if junit.exists():
        for test in ET.parse(junit).getroot().iter('testcase'):
            outcome = 'FAIL' if test.find('failure') is not None or test.find('error') is not None else 'SKIP' if test.find('skipped') is not None else 'PASS'
            child = test.find('failure') if outcome=='FAIL' else test.find('skipped')
            tests.append((test.attrib.get('name',''),outcome,child.attrib.get('message','') if child is not None else ''))
    lines = ['# SDDNet-v1 clean — execution evidence','',f'Architecture: **v1_clean**. Current stage: **{stage}**.',
        f'Formal completed epochs: **{completed}/100**. Smoke updates: **{smoke.get("optimizer_updates",0)}/20**.',
        'Smoke results, synthetic unit tests and diagnostic probes are not formal experiment accuracy.',
        'Old revision outputs are retained separately; old checkpoints are not valid v1_clean resumes.','',
        '## Environment','',f'Interpreter: `{environment.get("executable","NA")}`.',
        f'Torch {environment.get("torch","NA")}, CUDA {environment.get("torch_cuda","NA")}, cuDNN {environment.get("cudnn_version","NA")}.',
        f'GPU: {environment.get("gpu","NA")}; total VRAM bytes: {environment.get("vram_total_bytes","NA")}.',
        f'CUDA checks: {environment.get("cuda_test_status","NA: not executed")}. No dependencies were uninstalled.','',
        '## Model','',f'Measured parameters: {profile.get("parameters","NA: profile not run")} ({profile.get("parameters_M","NA")} M).',
        'For padded H8,W8: e0=[B,32,H8,W8], e1=[B,64,H8/2,W8/2], e2=[B,128,H8/4,W8/4], z=[B,256,H8/8,W8/8]; Fs/Fd=[B,128,H8/8,W8/8], c_hat=[B,1].',
        'Default forward returns only raw enhanced RGB at original size; return_latents=True additionally exposes Fs/Fd/c_hat. The same formula is used in train and eval.',
        'Hidden Conv/Linear: Kaiming normal for LeakyReLU(.1), fan_in, zero bias. Affine local/global final outputs, RGB residual head and illumination final Linear: Normal(0,.001), zero bias. Independent rho starts at log(.1/.9). Initialization only at construction.','',
        '## Objective and labels','',
        'L_rec=L1(y_raw,GT)+.2*(1-SSIM(clamp(y_raw),GT)); L_scene=TripletMarginLoss(.2,p=2,eps=1e-6,swap=False) on normalized GAP(Fs); L_illum=L1(c_hat,c_gt); L_cons=mean(max(1-channel_cosine(Fs_a,Fs_p),0)).',
        'L_total=L_rec+.1*L_scene+.05*L_illum+lambda_cons(progress)*L_cons; lambda_cons ramps 0→.02 over the first 5 epochs. c_gt=(level-1)/9, float32 [N,1], never a forward input.',
        'No unbounded Fd separation, arbitrary whole-feature spacing, orthogonality, hard frequency split, IN substitution or reconstruction stop-gradient is added. Such terms do not establish identifiability and may conflict with reconstruction or permit scale cheating.','',
        '## Actual automated tests','', '| Test | Status | Reason |','|---|---|---|']
    lines += [f'| {name} | {status} | {reason.replace(chr(10)," ")} |' for name,status,reason in tests] or ['| pending | NA | tests not executed |']
    lines += ['', '## Real-data short training','',f'Status: {smoke.get("status","NA: not executed")}.',
        f'Actual successful optimizer updates: {smoke.get("optimizer_updates","NA")}; attempted: {smoke.get("attempted_updates","NA")}; AMP overflows: {smoke.get("AMP_overflows","NA")}.',
        f'Peak training allocated bytes: {smoke.get("peak_train_allocated_bytes","NA")}; measured seconds: {smoke.get("training_seconds","NA")}.',
        'Full raw/weighted loss terms and weights are in smoke/updates.csv; diagnostics and interventions are in smoke/diagnostics.csv. These are not 20 formal epochs.',
        f'Limited validation probe: `{smoke.get("validation_probe","NA")}`. Scope: {smoke.get("validation_scope","not executed")}.', '',
        '## Sensitivity diagnostics','',
        'Fixed validation center crop, >=2 different levels of one aligned content plus another content. Identity and separate c_hat/Fd shuffles use temporary hooks and are restored. Differences can be small or zero and are reported as measured; these are sensitivity checks, not causal independence proofs.']
    if smoke.get('diagnostic'):
        lines += ['', '```json',json.dumps(smoke['diagnostic'],ensure_ascii=False,indent=2),'```']
    lines += ['', '## Inference profile','',f'Scope: {profile.get("scope","NA")}; precision: {profile.get("precision","NA")}.',
        f'Latency mean ms: {profile.get("latency_ms_mean","NA")}; FPS: {profile.get("FPS","NA")}; inference allocated peak bytes: {profile.get("peak_inference_allocated_bytes","NA")}.',
        'Fixed [1,3,400,600], 20 warmups, 100 timed synchronized CUDA forwards; IO/transfers/metrics excluded. Missing profiling metrics are NA with reason in profile JSON.','',
        '## Formal test results','']
    if summary:
        lines += ['| Dataset | Kind | N | PSNR_RGB | SSIM_RGB | LPIPS_Alex |','|---|---|---:|---:|---:|---:|']
        lines += [f'| {r["dataset"]} | {r["kind"]} | {r["count"]} | {r["PSNR_RGB"]} | {r["SSIM_RGB"]} | {r["LPIPS_Alex"]} |' for r in summary]
    else:
        lines += ['NA: v1_clean final test evaluation is not complete yet. No test accuracy numbers are fabricated.']
    lines += ['', 'MILL-only training; LOL is cross-dataset evaluation, never LOL-domain training. FP32, batch1, native RGB, outer clamp, no TTA/tiles/GT alignment/extra crop; float predictions before PNG quantization. SSIM pytorch-msssim 1.0.0, win11 sigma1.5 data_range1 K=(.01,.03), nonnegative=False. PSNR_RGB is not PSNR_L. LPIPS_Alex v0.1 is pretrained/frozen, inputs 2*x-1; failures are NA with reasons.', '',
        '## Commands','', '```bat','conda activate testmodel',
        'python -m scripts.run_pipeline --config configs/sddnet_v1.yaml --check-only',
        'python -m scripts.run_pipeline --config configs/sddnet_v1.yaml --smoke-test',
        'python -m scripts.run_pipeline --config configs/sddnet_v1.yaml',
        'python -m scripts.run_pipeline --config configs/sddnet_v1.yaml --resume outputs/sddnet_v1_clean_mill_100e/checkpoints/last.pth',
        'python -m scripts.run_pipeline --config configs/sddnet_v1.yaml --eval-only --checkpoint outputs/sddnet_v1_clean_mill_100e/checkpoints/best.pth','```',
        '', 'Commands run from project root. Model/initialization/objective settings are experimental designs, not previously validated claims.']
    if (directory/'log_recovery.json').exists():
        lines += ['', '## Interrupted CSV recovery', '', '```json',
                  json.dumps(read_json(directory/'log_recovery.json'),ensure_ascii=False,indent=2),'```',
                  'While a CSV is locked by a viewer, the newest complete snapshot is retained as *.pending.csv and read by the pipeline. Per-epoch records are also journaled in epoch_records/.']
    (directory/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    if history:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        figure, axes = plt.subplots(1,2,figsize=(10,4))
        epochs = [int(row['completed_epoch']) for row in history]
        axes[0].plot(epochs,[float(row['total']) if row.get('total') not in (None,'','NA') else float('nan') for row in history])
        axes[0].set(xlabel='completed epoch',ylabel='train total')
        axes[1].plot(epochs,[float(row['val_PSNR_RGB']) for row in history])
        axes[1].set(xlabel='completed epoch',ylabel='validation PSNR_RGB')
        figure.tight_layout()
        figure.savefig(directory/'curves.png',dpi=150)
        plt.close(figure)
