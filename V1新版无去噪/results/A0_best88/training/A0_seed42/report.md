# SDDNet-v1 controlled condition ablation — execution evidence

Architecture: **v1_clean**. Current stage: **train_only_complete**.
Formal completed epochs: **100/100**. Smoke updates: **0/20**.
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
| test_shapes_return_contract_prior_and_affine[shape0] | PASS |  |
| test_shapes_return_contract_prior_and_affine[shape1] | PASS |  |
| test_shapes_return_contract_prior_and_affine[shape2] | PASS |  |
| test_shapes_return_contract_prior_and_affine[shape3] | PASS |  |
| test_shapes_return_contract_prior_and_affine[shape4] | PASS |  |
| test_invalid_input_messages[x0-ValueError] | PASS |  |
| test_invalid_input_messages[x1-ValueError] | PASS |  |
| test_invalid_input_messages[x2-TypeError] | PASS |  |
| test_invalid_input_messages[x3-ValueError] | PASS |  |
| test_prior_is_computed_before_padding | PASS |  |
| test_exact_label_shape_triplet_formula_and_warmup | PASS |  |
| test_consistency_cosine_and_unclamped_l1_gradient | PASS |  |
| test_reconstruction_paths_and_hook_restoration | PASS |  |
| test_initialization_and_decay_groups | PASS |  |
| test_sampler_and_accumulation_tail | PASS |  |
| test_metadata_triplet_alignment_and_same_level_negative | PASS |  |
| test_metrics_properties_and_reference | PASS |  |
| test_pretrained_lpips_identical_image | PASS |  |
| test_run_lock | PASS |  |
| test_exact_affine_formulas_no_hidden_standard_gain[bounded] | PASS |  |
| test_exact_affine_formulas_no_hidden_standard_gain[standard] | PASS |  |
| test_formula_and_local_initialization_are_separate_factors | PASS |  |
| test_standard_private_decode_matches_public_forward_and_fixed_features[shape0] | PASS |  |
| test_standard_private_decode_matches_public_forward_and_fixed_features[shape1] | PASS |  |
| test_standard_private_decode_matches_public_forward_and_fixed_features[shape2] | PASS |  |
| test_predicted_gate_train_eval_and_level10_not_identity | PASS |  |
| test_semihard_legal_selection_hard_fallback_and_original_gradient | PASS |  |
| test_missing_negative_skips_and_easy_zero_is_valid | PASS |  |
| test_pk_positive_alignment_candidate_counts_and_sampler_resume | PASS |  |
| test_fixed_validation_donors_no_fixed_points_and_state_restoration | PASS |  |
| test_config_checkpoint_hash_factors_and_safe_old_weight_rejection | PASS |  |
| test_development_data_audit_skips_test_lol_and_detects_content_leak | PASS |  |
| test_scene_cluster_bootstrap_counts_clusters_not_levels | PASS |  |
| test_a0_metadata_statistics_do_not_change_original_loss_or_gradients | PASS |  |
| test_control_comparison_keeps_three_different_delta_definitions_separate | PASS |  |
| test_windows_cli_reconfigures_console_without_changing_parser | PASS |  |
| test_pk_nonfinite_failure_keeps_actual_image_identifiers | PASS |  |
| test_legacy_affine_hook_cannot_silently_leave_gate_condition_unchanged | PASS |  |
| test_train_only_cli_never_reads_final_data_or_lpips[True] | PASS |  |
| test_train_only_cli_never_reads_final_data_or_lpips[False] | PASS |  |
| test_train_only_resume_cli_and_conflicting_flags | PASS |  |
| test_existing_formal_results_rejected_before_writes | PASS |  |
| test_profile_disabled_does_not_load_or_construct | PASS |  |
| test_eval_only_profile_disabled_at_entry | PASS |  |
| test_compare_mixed_datasets_splits_and_same_names_are_separate | PASS |  |
| test_compare_validation_defaults_levels_content_and_nonfinite | PASS |  |
| test_compare_legacy_lol_na_labels_are_not_fabricated | PASS |  |
| test_a0_a1_only_formula_identity_diff_and_all_common_initial_weights_equal | PASS |  |
| test_best_validation_export_strict_own_best_full_count_and_metadata | PASS |  |
| test_calibration_read_only_hashes_labels_views_and_no_old_writes | PASS |  |
| test_locked_csv_retains_new_snapshot_and_commits_after_unlock | PASS |  |
| test_locked_non_log_destination_fails_explicitly | PASS |  |
| test_real_windows_file_sharing_lock | PASS |  |
| test_branch_norms_batch_one_cpu_transfer_without_item | PASS |  |
| test_amp_candidate_fuses_scale_and_skip_flag_in_one_transfer | PASS |  |
| test_unsupported_scaler_fails_explicitly_instead_of_guessing_skip | PASS |  |
| test_eval_only_rejects_before_any_output_data_or_inference[smoke] | PASS |  |
| test_eval_only_rejects_before_any_output_data_or_inference[None] | PASS |  |
| test_eval_only_rejects_before_any_output_data_or_inference[unknown] | PASS |  |
| test_eval_only_rejects_before_any_output_data_or_inference[] | PASS |  |
| test_eval_only_rejects_before_any_output_data_or_inference[Formal] | PASS |  |
| test_direct_evaluator_cannot_bypass_kind_guard[smoke] | PASS |  |
| test_direct_evaluator_cannot_bypass_kind_guard[None] | PASS |  |
| test_direct_evaluator_cannot_bypass_kind_guard[unknown] | PASS |  |
| test_formal_evaluation_runs_metrics_on_one_temporary_fixture | PASS |  |
| test_formal_guard_keeps_existing_metadata_checks[arch_revision-Legacy resume rejected] | PASS |  |
| test_formal_guard_keeps_existing_metadata_checks[contract-contract] | PASS |  |
| test_formal_guard_keeps_existing_metadata_checks[manifest_hashes-fingerprint] | PASS |  |
| test_formal_resume_and_generic_smoke_diagnostics_policy | PASS |  |
| test_cli_gradient_option_and_yaml_default[0] | PASS |  |
| test_cli_gradient_option_and_yaml_default[1] | PASS |  |
| test_cli_gradient_option_and_yaml_default[100] | PASS |  |
| test_actual_eval_cli_accepts_formal_checkpoint_and_resolves_diagnostic_override[None] | PASS |  |
| test_actual_eval_cli_accepts_formal_checkpoint_and_resolves_diagnostic_override[0] | PASS |  |
| test_actual_eval_cli_accepts_formal_checkpoint_and_resolves_diagnostic_override[1] | PASS |  |
| test_actual_eval_cli_accepts_formal_checkpoint_and_resolves_diagnostic_override[100] | PASS |  |
| test_smoke_rejection_preserves_existing_output_evidence | PASS |  |
| test_actual_resume_cli_keeps_formal_validation_and_mode | PASS |  |
| test_direct_formal_training_rejects_smoke_before_constructing_trainer | PASS |  |

## Real-data short training

Status: NA: not executed.
Actual successful optimizer updates: NA; attempted: NA; AMP overflows: NA.
Peak training allocated bytes: NA; measured seconds: NA.
Full raw/weighted loss terms and weights are in smoke/updates.csv; diagnostics and interventions are in smoke/diagnostics.csv. These are not 20 formal epochs.
Limited validation probe: `NA`. Scope: not executed.

## Sensitivity diagnostics

Fixed validation center crop, >=2 different levels of one aligned content plus another content. Identity and separate c_hat/Fd shuffles use temporary hooks and are restored. Differences can be small or zero and are reported as measured; these are sensitivity checks, not causal independence proofs.

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
