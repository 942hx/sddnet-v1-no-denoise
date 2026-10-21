"""Report this A0 run's actual records without running additional experiments."""
from utils.experiment import run_path, read_json
from utils.io import read_csv


def report(cfg, stage='formal_training', trainer=None):
    directory = run_path(cfg)
    history = read_csv(directory/'history.csv')
    summary = read_csv(directory/'metrics/summary.csv')
    completed = trainer.completed_epoch if trainer else int(history[-1]['completed_epoch']) if history else 0
    environment = read_json(directory/'environment.json') if (directory/'environment.json').exists() else {}
    parameters = sum(p.numel() for p in trainer.model.parameters()) if trainer else 'NA: not measured in this report'
    lines = [
        '# SDDNet-v1 A0 — execution evidence', '',
        f'Stage: {stage}; completed formal epochs: {completed}/100; measured parameters: {parameters}.',
        'Historical frozen best88 evidence is under results/A0_best88; this report reads only this run.', '',
        '## Environment', '',
        f'Interpreter: `{environment.get("executable", "NA")}`.',
        f'Torch {environment.get("torch", "NA")}; CUDA {environment.get("torch_cuda", "NA")}; cuDNN {environment.get("cudnn_version", "NA")}.',
        f'GPU: {environment.get("gpu", "NA")}.', '',
        '## Actual A0 configuration', '',
        f'Model: `{cfg["model"]}`; sampling: `{cfg["sampling"]}`; loss: `{cfg["loss"]}`.',
        'Bounded affine with learned rho, residual_gate=false, no independent denoising or wavelets.',
        'Normal forward consumes only low-light RGB and returns raw RGB; return_latents additionally exposes Fs/Fd/c_hat.',
        'L_rec=L1(raw,GT)+.2*(1-SSIM(clamp(raw),GT)); L_scene=TripletMarginLoss(.2,p=2,eps=1e-6,swap=False) on normalized GAP(Fs).',
        'L_illum=L1(c_hat,c_gt), c_gt=(level-1)/9; L_cons=mean(max(1-channel_cosine(Fs_a,Fs_p),0)).',
        'Total=L_rec+.1*L_scene+.05*L_illum+lambda_cons(progress)*L_cons; lambda_cons ramps 0 to .02 over the first 5 epochs.',
        'MILL train/validation only; validation selects best. Training does not automatically run test/LOL.', '',
        '## Recorded test results', '',
    ]
    status_path = directory/'metrics/evaluation_status.json'
    if status_path.exists():
        status = read_json(status_path)
        lines.append(f'Explicit scope: {status.get("datasets")}; limited_smoke={status.get("limited_smoke")}; failures={status.get("failures")}.')
    if summary:
        lines += ['| Dataset | Kind | N | PSNR_RGB | SSIM_RGB | LPIPS_Alex |', '|---|---|---:|---:|---:|---:|']
        lines += [f'| {r["dataset"]} | {r["kind"]} | {r["count"]} | {r["PSNR_RGB"]} | {r["SSIM_RGB"]} | {r["LPIPS_Alex"]} |' for r in summary]
    else:
        lines.append('NA: test evaluation has not been run in this output directory.')
    lines += ['',
        'FP32, batch1, native RGB, outer clamp, metrics before PNG quantization; no resize, tiles, TTA or GT alignment.',
        'PSNR_RGB is a per-image mean. SSIM: pytorch-msssim 1.0.0, win11, sigma1.5, data_range1, K=(.01,.03), nonnegative=False. Frozen LPIPS_Alex v0.1: unavailable metrics are NA with reasons.',
        'Validation and private condition interventions are distinct evidence; oracle/condition sensitivity does not establish physical or causal disentanglement.',
        'Existing LOL results have already been viewed; no untouched-test, universal improvement or SOTA claim is made.', '',
        '## Records and commands', '',
        'Training records: history.csv, train.log, tensorboard/, epoch_records/, coverage/, mining/. Config/environment describe this run.',
        'Best full validation is exported after training to validation/best/. Current CLI and limitations: project README.md.',
        '```text',
        'python -m scripts.train --config configs/a0.yaml --output outputs/NEW_RUN',
        'python -m scripts.evaluate --config configs/a0.yaml --dataset mill --output outputs/NEW_EVAL',
        'python -m scripts.infer --config configs/a0.yaml --input IMAGE --output outputs/NEW_INFER',
        '```',
    ]
    if (directory/'log_recovery.json').exists():
        import json
        lines += ['', '## Interrupted CSV recovery', '', '```json',
                  json.dumps(read_json(directory/'log_recovery.json'), ensure_ascii=False, indent=2), '```',
                  'Newest complete locked CSV snapshots use *.pending.csv; per-epoch records are journaled in epoch_records/.']
    (directory/'report.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    if history:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        figure, axes = plt.subplots(1, 2, figsize=(10, 4))
        epochs = [int(row['completed_epoch']) for row in history]
        axes[0].plot(epochs, [float(row['total']) if row.get('total') not in (None, '', 'NA') else float('nan') for row in history])
        axes[0].set(xlabel='completed epoch', ylabel='train total')
        axes[1].plot(epochs, [float(row['val_PSNR_RGB']) for row in history])
        axes[1].set(xlabel='completed epoch', ylabel='validation PSNR_RGB')
        figure.tight_layout()
        figure.savefig(directory/'curves.png', dpi=150)
        plt.close(figure)
