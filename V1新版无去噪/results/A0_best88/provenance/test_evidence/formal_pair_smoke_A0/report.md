# SDDNet-v1 controlled condition ablation — execution evidence

Architecture: **v1_clean**. Current stage: **smoke_complete**.
Formal completed epochs: **0/100**. Smoke updates: **20/20**.
Smoke results, synthetic unit tests and diagnostic probes are not formal experiment accuracy.
Old revision outputs are retained separately; old checkpoints are not valid v1_clean resumes.

## Environment

Interpreter: `C:\software\anaconda3\envs\testmodel\python.exe`.
Torch 2.5.1+cu121, CUDA 12.1, cuDNN 90100.
GPU: NVIDIA GeForce RTX 4060 Laptop GPU; total VRAM bytes: 8585216000.
CUDA checks: PASS: real CUDA matrix multiplication and convolution backward. No dependencies were uninstalled.

## Model

Measured parameters: NA: profile not run (NA M).
For padded H8,W8: e0=[B,32,H8,W8], e1=[B,64,H8/2,W8/2], e2=[B,128,H8/4,W8/4], z=[B,256,H8/8,W8/8]; Fs/Fd=[B,128,H8/8,W8/8], c_hat=[B,1].
Default forward returns only raw enhanced RGB at original size; return_latents=True additionally exposes Fs/Fd/c_hat. The same formula is used in train and eval.
Hidden Conv/Linear: Kaiming normal for LeakyReLU(.1), fan_in, zero bias. Affine local/global final outputs, RGB residual head and illumination final Linear: Normal(0,.001), zero bias. Independent rho starts at log(.1/.9). Initialization only at construction.

## Objective and labels

L_rec=L1(y_raw,GT)+.2*(1-SSIM(clamp(y_raw),GT)); L_scene=TripletMarginLoss(.2,p=2,eps=1e-6,swap=False) on normalized GAP(Fs); L_illum=L1(c_hat,c_gt); L_cons=mean(max(1-channel_cosine(Fs_a,Fs_p),0)).
L_total=L_rec+.1*L_scene+.05*L_illum+lambda_cons(progress)*L_cons; lambda_cons ramps 0→.02 over the first 5 epochs. c_gt=(level-1)/9, float32 [N,1], never a forward input.
No unbounded Fd separation, arbitrary whole-feature spacing, orthogonality, hard frequency split, IN substitution or reconstruction stop-gradient is added. Such terms do not establish identifiability and may conflict with reconstruction or permit scale cheating.

## Actual automated tests

| Test | Status | Reason |
|---|---|---|
| pending | NA | tests not executed |

## Real-data short training

Status: PASS.
Actual successful optimizer updates: 20; attempted: 21; AMP overflows: 1.
Peak training allocated bytes: 1048394752; measured seconds: 4.769708499999998.
Full raw/weighted loss terms and weights are in smoke/updates.csv; diagnostics and interventions are in smoke/diagnostics.csv. These are not 20 formal epochs.
Limited validation probe: `{'PSNR_RGB': 20.043830293648032, 'SSIM_RGB': 0.7498388191064199, 'count': 6, 'per_image': [{'filename': 'Test1-1.png', 'content_id': 'Test1', 'group_id': '', 'level': 1.0, 'split': 'validation', 'PSNR_RGB': 12.295430360479285, 'SSIM_RGB': 0.32768577337265015, 'input_PSNR_RGB': 11.213351976622603, 'input_SSIM_RGB': 0.20235739648342133, 'delta_psnr_vs_input': 1.0820783838566825, 'delta_psnr_vs_control': 'NA', 'c_hat': 0.5000673532485962, 'illumination_MAE': 0.5000673532485962, 'gate_mean': 1.0, 'delta_abs_mean': 0.030428504571318626, 'gated_delta_abs_mean': 0.030428504571318626, 'rgb_mean_signed_error': -0.22461757063865662, 'channel_mean_error_abs': 0.22461757063865662, 'saturation_fraction': 0.0}, {'filename': 'Test1-10.png', 'content_id': 'Test1', 'group_id': '', 'level': 10.0, 'split': 'validation', 'PSNR_RGB': 18.25424292001958, 'SSIM_RGB': 0.9032102823257446, 'input_PSNR_RGB': 35.42041695770544, 'input_SSIM_RGB': 0.9727212190628052, 'delta_psnr_vs_input': -17.166174037685863, 'delta_psnr_vs_control': 'NA', 'c_hat': 0.5008722543716431, 'illumination_MAE': 0.49912774562835693, 'gate_mean': 1.0, 'delta_abs_mean': 0.13013987243175507, 'gated_delta_abs_mean': 0.13013987243175507, 'rgb_mean_signed_error': 0.11502078175544739, 'channel_mean_error_abs': 0.11502079665660858, 'saturation_fraction': 0.014795833267271519}, {'filename': 'Test1-2.png', 'content_id': 'Test1', 'group_id': '', 'level': 2.0, 'split': 'validation', 'PSNR_RGB': 15.82175251301409, 'SSIM_RGB': 0.6420822143554688, 'input_PSNR_RGB': 13.308374439610525, 'input_SSIM_RGB': 0.4690920114517212, 'delta_psnr_vs_input': 2.5133780734035653, 'delta_psnr_vs_control': 'NA', 'c_hat': 0.5002437829971313, 'illumination_MAE': 0.38913267188602024, 'gate_mean': 1.0, 'delta_abs_mean': 0.05267609655857086, 'gated_delta_abs_mean': 0.05267609655857086, 'rgb_mean_signed_error': -0.14688588678836823, 'channel_mean_error_abs': 0.14688587188720703, 'saturation_fraction': 2.7777778086601757e-06}, {'filename': 'Test1-3.png', 'content_id': 'Test1', 'group_id': '', 'level': 3.0, 'split': 'validation', 'PSNR_RGB': 19.603785608380008, 'SSIM_RGB': 0.8081428408622742, 'input_PSNR_RGB': 15.176193150366856, 'input_SSIM_RGB': 0.6541950106620789, 'delta_psnr_vs_input': 4.427592458013152, 'delta_psnr_vs_control': 'NA', 'c_hat': 0.5003766417503357, 'illumination_MAE': 0.2781544195281135, 'gate_mean': 1.0, 'delta_abs_mean': 0.06889339536428452, 'gated_delta_abs_mean': 0.06889339536428452, 'rgb_mean_signed_error': -0.09096668660640717, 'channel_mean_error_abs': 0.09096666425466537, 'saturation_fraction': 1.1111111234640703e-05}, {'filename': 'Test1-4.png', 'content_id': 'Test1', 'group_id': '', 'level': 4.0, 'split': 'validation', 'PSNR_RGB': 23.938115479117293, 'SSIM_RGB': 0.8897831439971924, 'input_PSNR_RGB': 16.88867400272089, 'input_SSIM_RGB': 0.7715363502502441, 'delta_psnr_vs_input': 7.049441476396403, 'delta_psnr_vs_control': 'NA', 'c_hat': 0.500476598739624, 'illumination_MAE': 0.1671432654062907, 'gate_mean': 1.0, 'delta_abs_mean': 0.0809902474284172, 'gated_delta_abs_mean': 0.0809902474284172, 'rgb_mean_signed_error': -0.0496220588684082, 'channel_mean_error_abs': 0.04962204024195671, 'saturation_fraction': 2.6388888727524318e-05}, {'filename': 'Test1-5.png', 'content_id': 'Test1', 'group_id': '', 'level': 5.0, 'split': 'validation', 'PSNR_RGB': 30.349654880877935, 'SSIM_RGB': 0.9281286597251892, 'input_PSNR_RGB': 19.07081275774725, 'input_SSIM_RGB': 0.8566431999206543, 'delta_psnr_vs_input': 11.278842123130683, 'delta_psnr_vs_control': 'NA', 'c_hat': 0.5005747675895691, 'illumination_MAE': 0.05613032314512467, 'gate_mean': 1.0, 'delta_abs_mean': 0.09326740354299545, 'gated_delta_abs_mean': 0.09326740354299545, 'rgb_mean_signed_error': -0.008075475692749023, 'channel_mean_error_abs': 0.008075490593910217, 'saturation_fraction': 4.166666622040793e-05}], 'per_level': [{'level': 1, 'count': 1, 'scope': 'limited smoke validation', 'PSNR_RGB': 12.295430360479285, 'SSIM_RGB': 0.32768577337265015, 'input_PSNR_RGB': 11.213351976622603, 'input_SSIM_RGB': 0.20235739648342133, 'illumination_MAE': 0.5000673532485962, 'gate_mean': 1.0, 'delta_abs_mean': 0.030428504571318626, 'gated_delta_abs_mean': 0.030428504571318626, 'rgb_mean_signed_error': -0.22461757063865662, 'channel_mean_error_abs': 0.22461757063865662, 'saturation_fraction': 0.0, 'delta_psnr_vs_input': 1.0820783838566825, 'delta_psnr_vs_input_median': 1.0820783838566825, 'improvement_image_fraction': 1.0, 'decline_image_fraction': 0.0, 'cluster_bootstrap_ci95_low': 'NA', 'cluster_bootstrap_ci95_high': 'NA', 'paired_reason': 'Fewer than two independent content/group clusters; CI is NA.', 'bootstrap_unit': '(dataset, split, verified group_id or full content_id)', 'bootstrap_clusters': 1, 'bootstrap_seed': 42}, {'level': 2, 'count': 1, 'scope': 'limited smoke validation', 'PSNR_RGB': 15.82175251301409, 'SSIM_RGB': 0.6420822143554688, 'input_PSNR_RGB': 13.308374439610525, 'input_SSIM_RGB': 0.4690920114517212, 'illumination_MAE': 0.38913267188602024, 'gate_mean': 1.0, 'delta_abs_mean': 0.05267609655857086, 'gated_delta_abs_mean': 0.05267609655857086, 'rgb_mean_signed_error': -0.14688588678836823, 'channel_mean_error_abs': 0.14688587188720703, 'saturation_fraction': 2.7777778086601757e-06, 'delta_psnr_vs_input': 2.5133780734035653, 'delta_psnr_vs_input_median': 2.5133780734035653, 'improvement_image_fraction': 1.0, 'decline_image_fraction': 0.0, 'cluster_bootstrap_ci95_low': 'NA', 'cluster_bootstrap_ci95_high': 'NA', 'paired_reason': 'Fewer than two independent content/group clusters; CI is NA.', 'bootstrap_unit': '(dataset, split, verified group_id or full content_id)', 'bootstrap_clusters': 1, 'bootstrap_seed': 42}, {'level': 3, 'count': 1, 'scope': 'limited smoke validation', 'PSNR_RGB': 19.603785608380008, 'SSIM_RGB': 0.8081428408622742, 'input_PSNR_RGB': 15.176193150366856, 'input_SSIM_RGB': 0.6541950106620789, 'illumination_MAE': 0.2781544195281135, 'gate_mean': 1.0, 'delta_abs_mean': 0.06889339536428452, 'gated_delta_abs_mean': 0.06889339536428452, 'rgb_mean_signed_error': -0.09096668660640717, 'channel_mean_error_abs': 0.09096666425466537, 'saturation_fraction': 1.1111111234640703e-05, 'delta_psnr_vs_input': 4.427592458013152, 'delta_psnr_vs_input_median': 4.427592458013152, 'improvement_image_fraction': 1.0, 'decline_image_fraction': 0.0, 'cluster_bootstrap_ci95_low': 'NA', 'cluster_bootstrap_ci95_high': 'NA', 'paired_reason': 'Fewer than two independent content/group clusters; CI is NA.', 'bootstrap_unit': '(dataset, split, verified group_id or full content_id)', 'bootstrap_clusters': 1, 'bootstrap_seed': 42}, {'level': 4, 'count': 1, 'scope': 'limited smoke validation', 'PSNR_RGB': 23.938115479117293, 'SSIM_RGB': 0.8897831439971924, 'input_PSNR_RGB': 16.88867400272089, 'input_SSIM_RGB': 0.7715363502502441, 'illumination_MAE': 0.1671432654062907, 'gate_mean': 1.0, 'delta_abs_mean': 0.0809902474284172, 'gated_delta_abs_mean': 0.0809902474284172, 'rgb_mean_signed_error': -0.0496220588684082, 'channel_mean_error_abs': 0.04962204024195671, 'saturation_fraction': 2.6388888727524318e-05, 'delta_psnr_vs_input': 7.049441476396403, 'delta_psnr_vs_input_median': 7.049441476396403, 'improvement_image_fraction': 1.0, 'decline_image_fraction': 0.0, 'cluster_bootstrap_ci95_low': 'NA', 'cluster_bootstrap_ci95_high': 'NA', 'paired_reason': 'Fewer than two independent content/group clusters; CI is NA.', 'bootstrap_unit': '(dataset, split, verified group_id or full content_id)', 'bootstrap_clusters': 1, 'bootstrap_seed': 42}, {'level': 5, 'count': 1, 'scope': 'limited smoke validation', 'PSNR_RGB': 30.349654880877935, 'SSIM_RGB': 0.9281286597251892, 'input_PSNR_RGB': 19.07081275774725, 'input_SSIM_RGB': 0.8566431999206543, 'illumination_MAE': 0.05613032314512467, 'gate_mean': 1.0, 'delta_abs_mean': 0.09326740354299545, 'gated_delta_abs_mean': 0.09326740354299545, 'rgb_mean_signed_error': -0.008075475692749023, 'channel_mean_error_abs': 0.008075490593910217, 'saturation_fraction': 4.166666622040793e-05, 'delta_psnr_vs_input': 11.278842123130683, 'delta_psnr_vs_input_median': 11.278842123130683, 'improvement_image_fraction': 1.0, 'decline_image_fraction': 0.0, 'cluster_bootstrap_ci95_low': 'NA', 'cluster_bootstrap_ci95_high': 'NA', 'paired_reason': 'Fewer than two independent content/group clusters; CI is NA.', 'bootstrap_unit': '(dataset, split, verified group_id or full content_id)', 'bootstrap_clusters': 1, 'bootstrap_seed': 42}, {'level': 6, 'count': 0, 'scope': 'limited smoke validation', 'PSNR_RGB': 'NA', 'SSIM_RGB': 'NA', 'input_PSNR_RGB': 'NA', 'input_SSIM_RGB': 'NA', 'illumination_MAE': 'NA', 'gate_mean': 'NA', 'delta_abs_mean': 'NA', 'gated_delta_abs_mean': 'NA', 'rgb_mean_signed_error': 'NA', 'channel_mean_error_abs': 'NA', 'saturation_fraction': 'NA', 'delta_psnr_vs_input': 'NA', 'paired_reason': 'no images'}, {'level': 7, 'count': 0, 'scope': 'limited smoke validation', 'PSNR_RGB': 'NA', 'SSIM_RGB': 'NA', 'input_PSNR_RGB': 'NA', 'input_SSIM_RGB': 'NA', 'illumination_MAE': 'NA', 'gate_mean': 'NA', 'delta_abs_mean': 'NA', 'gated_delta_abs_mean': 'NA', 'rgb_mean_signed_error': 'NA', 'channel_mean_error_abs': 'NA', 'saturation_fraction': 'NA', 'delta_psnr_vs_input': 'NA', 'paired_reason': 'no images'}, {'level': 8, 'count': 0, 'scope': 'limited smoke validation', 'PSNR_RGB': 'NA', 'SSIM_RGB': 'NA', 'input_PSNR_RGB': 'NA', 'input_SSIM_RGB': 'NA', 'illumination_MAE': 'NA', 'gate_mean': 'NA', 'delta_abs_mean': 'NA', 'gated_delta_abs_mean': 'NA', 'rgb_mean_signed_error': 'NA', 'channel_mean_error_abs': 'NA', 'saturation_fraction': 'NA', 'delta_psnr_vs_input': 'NA', 'paired_reason': 'no images'}, {'level': 9, 'count': 0, 'scope': 'limited smoke validation', 'PSNR_RGB': 'NA', 'SSIM_RGB': 'NA', 'input_PSNR_RGB': 'NA', 'input_SSIM_RGB': 'NA', 'illumination_MAE': 'NA', 'gate_mean': 'NA', 'delta_abs_mean': 'NA', 'gated_delta_abs_mean': 'NA', 'rgb_mean_signed_error': 'NA', 'channel_mean_error_abs': 'NA', 'saturation_fraction': 'NA', 'delta_psnr_vs_input': 'NA', 'paired_reason': 'no images'}, {'level': 10, 'count': 1, 'scope': 'limited smoke validation', 'PSNR_RGB': 18.25424292001958, 'SSIM_RGB': 0.9032102823257446, 'input_PSNR_RGB': 35.42041695770544, 'input_SSIM_RGB': 0.9727212190628052, 'illumination_MAE': 0.49912774562835693, 'gate_mean': 1.0, 'delta_abs_mean': 0.13013987243175507, 'gated_delta_abs_mean': 0.13013987243175507, 'rgb_mean_signed_error': 0.11502078175544739, 'channel_mean_error_abs': 0.11502079665660858, 'saturation_fraction': 0.014795833267271519, 'delta_psnr_vs_input': -17.166174037685863, 'delta_psnr_vs_input_median': -17.166174037685863, 'improvement_image_fraction': 0.0, 'decline_image_fraction': 1.0, 'cluster_bootstrap_ci95_low': 'NA', 'cluster_bootstrap_ci95_high': 'NA', 'paired_reason': 'Fewer than two independent content/group clusters; CI is NA.', 'bootstrap_unit': '(dataset, split, verified group_id or full content_id)', 'bootstrap_clusters': 1, 'bootstrap_seed': 42}]}`. Scope: first 6 validation images; not a formal result.

## Sensitivity diagnostics

Fixed validation center crop, >=2 different levels of one aligned content plus another content. Identity and separate c_hat/Fd shuffles use temporary hooks and are restored. Differences can be small or zero and are reported as measured; these are sensitivity checks, not causal independence proofs.

```json
{
  "scope": "fixed validation center crop; not formal full-image metrics",
  "probe_images": 3,
  "bottleneck_affine_a": 0.09998109191656113,
  "bottleneck_affine_gamma_mean": -0.0022419756278395653,
  "bottleneck_affine_gamma_abs_mean": 0.003347689053043723,
  "bottleneck_affine_beta_mean": 0.0003285505226813257,
  "bottleneck_affine_beta_abs_mean": 0.0038072436582297087,
  "bottleneck_affine_tanh_saturation_fraction": 0.0,
  "skip_affine2_a": 0.10007398575544357,
  "skip_affine2_gamma_mean": -0.00032660141005180776,
  "skip_affine2_gamma_abs_mean": 0.0013323421590030193,
  "skip_affine2_beta_mean": 0.00018077969434671104,
  "skip_affine2_beta_abs_mean": 0.002705579623579979,
  "skip_affine2_tanh_saturation_fraction": 0.0,
  "skip_affine1_a": 0.10007785260677338,
  "skip_affine1_gamma_mean": 0.00021888874471187592,
  "skip_affine1_gamma_abs_mean": 0.0013934706803411245,
  "skip_affine1_beta_mean": 0.0009422128787264228,
  "skip_affine1_beta_abs_mean": 0.0030118697322905064,
  "skip_affine1_tanh_saturation_fraction": 0.0,
  "skip_affine0_a": 0.10008621960878372,
  "skip_affine0_gamma_mean": 0.0002358724595978856,
  "skip_affine0_gamma_abs_mean": 0.0014777532778680325,
  "skip_affine0_beta_mean": 6.437121191993356e-05,
  "skip_affine0_beta_abs_mean": 0.0025390349328517914,
  "skip_affine0_tanh_saturation_fraction": 0.0,
  "Fs_mean": 0.019687868654727936,
  "Fs_std": 0.11379443109035492,
  "Fs_norm": 36.208553314208984,
  "Fd_mean": 0.05355873703956604,
  "Fd_std": 0.14266903698444366,
  "Fd_norm": 47.77983856201172,
  "embedding_variance_across_contents": 0.002004165668040514,
  "same_content_cross_level_distance": 0.40185025334358215,
  "different_content_same_level_distance": 1.0129821300506592,
  "c_hat_std": 0.00039546401239931583,
  "illumination_MAE": 0.49977612495422363,
  "median_constant_MAE": 0.48148149251937866,
  "y_raw_out_of_range_fraction": 0.0038655598182231188,
  "probe_PSNR_normal": 13.179693150126672,
  "identity_raw_output_MAE": 0.0019724711310118437,
  "identity_PSNR_delta": 0.01868507713182055,
  "shuffle_c_raw_output_MAE": 1.7508427845314145e-06,
  "shuffle_c_PSNR_delta": 8.341274252821052e-05,
  "shuffle_fd_raw_output_MAE": 0.0003370656631886959,
  "shuffle_fd_PSNR_delta": 0.01511443568881532,
  "interventions_restored_exactly": true,
  "structure_branch_grad_norm_rec_only": 0.01157731469720602,
  "degradation_branch_grad_norm_rec_only": 0.0005134070524945855,
  "illumination_head_grad_norm_rec_only": 0.0005386293632909656,
  "interpretation": "Sensitivity/dependency checks only; not a causal disentanglement proof.",
  "grad_norm_samples": 0
}
```

## Inference profile

Scope: NA; precision: NA.
Latency mean ms: NA; FPS: NA; inference allocated peak bytes: NA.
Fixed [1,3,400,600], 20 warmups, 100 timed synchronized CUDA forwards; IO/transfers/metrics excluded. Missing profiling metrics are NA with reason in profile JSON.

## Formal test results

NA: v1_clean final test evaluation is not complete yet. No test accuracy numbers are fabricated.

MILL-only training; LOL is cross-dataset evaluation, never LOL-domain training. FP32, batch1, native RGB, outer clamp, no TTA/tiles/GT alignment/extra crop; float predictions before PNG quantization. SSIM pytorch-msssim 1.0.0, win11 sigma1.5 data_range1 K=(.01,.03), nonnegative=False. PSNR_RGB is not PSNR_L. LPIPS_Alex v0.1 is pretrained/frozen, inputs 2*x-1; failures are NA with reasons.

## Commands

```bat
conda activate testmodel
python -m scripts.run_pipeline --config configs/sddnet_v1.yaml --check-only
python -m scripts.run_pipeline --config configs/sddnet_v1.yaml --smoke-test
python -m scripts.run_pipeline --config configs/sddnet_v1.yaml
python -m scripts.run_pipeline --config configs/sddnet_v1.yaml --resume outputs/sddnet_v1_clean_mill_100e/checkpoints/last.pth
python -m scripts.run_pipeline --config configs/sddnet_v1.yaml --eval-only --checkpoint outputs/sddnet_v1_clean_mill_100e/checkpoints/best.pth
```

Commands run from project root. Model/initialization/objective settings are experimental designs, not previously validated claims.

## Actual variant configuration

Variant: A0; model: `{'affine_mode': 'bounded', 'local_init_std': 0.001, 'residual_gate': False, 'g_min': 0.25}`; sampling: `{'mode': 'triplet'}`.
Miner: fixed; accumulation: 2.
For standard FiLM, gamma=1+delta_gamma, beta=raw_beta; there is no rho/tanh/sigmoid gain. The earlier bounded formula only describes A0. Local std is configured separately; global/head small nonzero initialization is unchanged.
The gate is disabled by default. If enabled it uses the same predicted c_hat, without detach, in training/validation/testing; g_min is not a claim that level10 equals GT.
Fixed-content diagnostics use private encode/decode, CPU-cached validation donor Fd/c, no fixed-point permutations; neutral affine preserves the gate, and gate-only/all-condition-off are separate.
Full image validation and crop interventions are separate CSVs. Raw output MAE and clamped output MAE are separate; quality drop=correct PSNR minus replaced-condition PSNR.
Existing LOLv1/LOLv2 test results have already been viewed. This task does not claim they are untouched or use them to select hyperparameters. This acceptance run reads only MILL train/validation images.
No complete ablation, statistical gain, or guaranteed PSNR improvement is claimed. See reports/condition_ablation_audit.md for actual acceptance status.
