### Training runs

| run | task | encoder | settings | rollouts | transitions | val err (cm) | val err moved (cm) | rel err moved | no-motion baseline (cm) | MPC rollout success | hours |
|---|---|---|---|---|---|---|---|---|---|---|---|
| ball | ball | pointnet | 64 | 24 | 3478 | 4.46 | 8.03 | 0.39 | 8.13 | 52% | 0.7 |
| block | block | pointnet | 64 | 24 | 3734 | 2.72 | 4.71 | 0.33 | 5.80 | 70% | 0.7 |
| block_big | block | pointnet | 128 | 56 | 8642 | 2.34 | 4.44 | 0.33 | 5.09 | 74% | 1.7 |

### Test-time latent inference (unseen physics)

Prediction error (cm) on held-out pushes of new environments; `moved` = pushes that moved the object > 5 mm.

| checkpoint | k probes | envs | mean latent | fitted on k | oracle | mean (moved) | fitted (moved) | oracle (moved) |
|---|---|---|---|---|---|---|---|---|
| ball/ablation_encoders/cross_attn_s0 | 1 | 56 | 5.26 | 4.80 | 2.06 | 8.43 | 8.00 | 3.82 |
| ball/ablation_encoders/cross_attn_s0 | 2 | 56 | 5.26 | 4.27 | 2.06 | 8.43 | 6.75 | 3.82 |
| ball/ablation_encoders/cross_attn_s0 | 4 | 55 | 5.10 | 3.21 | 2.06 | 8.24 | 5.26 | 3.88 |
| ball/ablation_encoders/cross_attn_s0 | 8 | 51 | 4.55 | 2.87 | 1.71 | 7.14 | 4.95 | 3.22 |
| ball/ablation_image_vs_pc/image_cls_s0 | 1 | 55 | 5.72 | 5.21 | 1.69 | 7.97 | 7.35 | 2.43 |
| ball/ablation_image_vs_pc/image_cls_s0 | 2 | 54 | 5.57 | 4.88 | 1.70 | 7.77 | 6.90 | 2.46 |
| ball/ablation_image_vs_pc/image_cls_s0 | 4 | 49 | 4.90 | 3.52 | 1.37 | 6.93 | 5.21 | 2.04 |
| ball/ablation_image_vs_pc/image_cls_s0 | 8 | 42 | 4.55 | 2.48 | 1.06 | 6.17 | 3.57 | 1.53 |
| ball/ablation_repr/image_af_cls_s0 | 1 | 53 | 4.41 | 4.04 | 1.89 | 6.99 | 5.85 | 3.07 |
| ball/ablation_repr/image_af_cls_s0 | 2 | 51 | 3.94 | 3.33 | 1.60 | 6.16 | 4.48 | 2.54 |
| ball/ablation_repr/image_af_cls_s0 | 4 | 48 | 3.76 | 3.23 | 1.49 | 6.03 | 3.91 | 2.41 |
| ball/ablation_repr/image_af_cls_s0 | 8 | 47 | 3.41 | 2.23 | 1.09 | 5.51 | 2.82 | 1.82 |
| ball/ablation_repr/image_af_cnn_s0 | 1 | 53 | 3.94 | 3.80 | 2.35 | 6.30 | 5.33 | 3.71 |
| ball/ablation_repr/image_af_cnn_s0 | 2 | 52 | 3.83 | 3.50 | 2.34 | 6.07 | 5.06 | 3.74 |
| ball/ablation_repr/image_af_cnn_s0 | 4 | 49 | 3.42 | 3.07 | 1.97 | 5.47 | 4.33 | 3.14 |
| ball/ablation_repr/image_af_cnn_s0 | 8 | 47 | 3.25 | 2.73 | 1.81 | 5.18 | 3.83 | 2.90 |
| ball/ensemble_wide | 1 | 50 | 3.84 | 3.05 | 0.88 | 6.07 | 4.69 | 1.49 |
| ball/ensemble_wide | 2 | 50 | 3.84 | 2.97 | 0.88 | 6.07 | 4.50 | 1.49 |
| ball/ensemble_wide | 4 | 47 | 3.48 | 2.15 | 0.61 | 5.58 | 3.19 | 1.01 |
| ball/ensemble_wide | 8 | 44 | 3.37 | 1.31 | 0.55 | 5.41 | 1.97 | 0.88 |
| ball | 1 | 54 | 4.46 | 4.27 | 1.17 | 6.90 | 4.73 | 2.12 |
| ball | 2 | 54 | 4.46 | 3.23 | 1.17 | 6.90 | 4.27 | 2.12 |
| ball | 4 | 53 | 4.55 | 2.58 | 1.19 | 6.90 | 3.68 | 2.12 |
| ball | 8 | 48 | 4.18 | 1.82 | 0.98 | 6.40 | 2.82 | 1.68 |
| block/ablation_arch/pointnet_wide_s0 | 1 | 61 | 3.32 | 2.49 | 1.37 | 4.89 | 3.83 | 2.29 |
| block/ablation_arch/pointnet_wide_s0 | 2 | 61 | 3.32 | 2.35 | 1.37 | 4.89 | 3.51 | 2.29 |
| block/ablation_arch/pointnet_wide_s0 | 4 | 61 | 3.32 | 2.29 | 1.37 | 4.89 | 3.50 | 2.29 |
| block/ablation_arch/pointnet_wide_s0 | 8 | 58 | 2.83 | 1.68 | 0.84 | 4.05 | 2.44 | 1.36 |
| block/ablation_image_vs_pc/image_cls_s0 | 1 | 60 | 3.09 | 3.05 | 1.54 | 4.93 | 4.75 | 2.79 |
| block/ablation_image_vs_pc/image_cls_s0 | 2 | 60 | 3.09 | 3.00 | 1.54 | 4.93 | 4.89 | 2.79 |
| block/ablation_image_vs_pc/image_cls_s0 | 4 | 59 | 3.12 | 2.91 | 1.56 | 4.92 | 4.86 | 2.79 |
| block/ablation_image_vs_pc/image_cls_s0 | 8 | 59 | 3.12 | 2.74 | 1.56 | 4.92 | 4.65 | 2.79 |
| block/ablation_repr/image_af_cls_s0 | 1 | 61 | 3.62 | 4.00 | 2.28 | 5.33 | 6.40 | 4.04 |
| block/ablation_repr/image_af_cls_s0 | 2 | 61 | 3.62 | 3.35 | 2.28 | 5.33 | 5.55 | 4.04 |
| block/ablation_repr/image_af_cls_s0 | 4 | 61 | 3.62 | 2.95 | 2.28 | 5.33 | 4.91 | 4.04 |
| block/ablation_repr/image_af_cls_s0 | 8 | 58 | 3.34 | 2.52 | 1.97 | 4.68 | 4.14 | 3.42 |
| block/ablation_repr/image_af_cnn_s0 | 1 | 61 | 4.38 | 4.37 | 2.96 | 5.51 | 5.75 | 4.25 |
| block/ablation_repr/image_af_cnn_s0 | 2 | 60 | 4.34 | 4.00 | 2.90 | 5.46 | 5.37 | 4.19 |
| block/ablation_repr/image_af_cnn_s0 | 4 | 59 | 4.40 | 3.88 | 2.94 | 5.50 | 5.15 | 4.21 |
| block/ablation_repr/image_af_cnn_s0 | 8 | 57 | 4.25 | 3.36 | 2.79 | 5.21 | 4.50 | 3.97 |
| block/ablation_repr/image_cnn_s0 | 1 | 57 | 3.63 | 3.20 | 2.34 | 4.95 | 4.96 | 3.65 |
| block/ablation_repr/image_cnn_s0 | 2 | 57 | 3.63 | 3.26 | 2.34 | 4.95 | 4.92 | 3.65 |
| block/ablation_repr/image_cnn_s0 | 4 | 57 | 3.63 | 3.11 | 2.34 | 4.95 | 4.79 | 3.65 |
| block/ablation_repr/image_cnn_s0 | 8 | 54 | 3.67 | 3.09 | 2.34 | 4.94 | 4.74 | 3.62 |
| block/ensemble_wide | 1 | 63 | 3.67 | 2.91 | 1.51 | 5.32 | 3.97 | 2.30 |
| block/ensemble_wide | 2 | 62 | 3.70 | 2.77 | 1.53 | 5.35 | 3.84 | 2.33 |
| block/ensemble_wide | 4 | 60 | 3.52 | 2.43 | 1.43 | 5.13 | 3.57 | 2.20 |
| block/ensemble_wide | 8 | 57 | 3.60 | 2.54 | 1.46 | 5.19 | 3.52 | 2.21 |
| block | 1 | 63 | 4.18 | 3.44 | 1.86 | 5.28 | 4.47 | 2.60 |
| block | 2 | 62 | 4.20 | 3.33 | 1.88 | 5.28 | 4.32 | 2.61 |
| block | 4 | 61 | 4.21 | 3.06 | 1.88 | 5.28 | 4.02 | 2.61 |
| block | 8 | 59 | 3.88 | 2.85 | 1.58 | 4.74 | 3.64 | 2.15 |
| block_big/ensemble_wide | 1 | 64 | 3.51 | 2.41 | 1.26 | 5.02 | 3.41 | 1.86 |
| block_big/ensemble_wide | 2 | 63 | 3.20 | 1.97 | 0.94 | 4.54 | 2.77 | 1.39 |
| block_big/ensemble_wide | 4 | 63 | 3.20 | 1.81 | 0.94 | 4.54 | 2.59 | 1.39 |
| block_big/ensemble_wide | 8 | 61 | 2.96 | 1.54 | 0.74 | 4.23 | 2.22 | 1.09 |
| block_big | 1 | 62 | 3.64 | 3.24 | 1.72 | 5.63 | 4.80 | 2.90 |
| block_big | 2 | 62 | 3.64 | 2.88 | 1.72 | 5.63 | 4.37 | 2.90 |
| block_big | 4 | 61 | 3.38 | 3.10 | 1.48 | 5.19 | 4.53 | 2.48 |
| block_big | 8 | 58 | 3.27 | 2.60 | 1.36 | 4.95 | 3.87 | 2.24 |

### Goal reaching with MPC (unseen physics)

`hard` = goal 15-30 cm away and only 2 pushes.

| checkpoint | task | envs | start→goal (cm) | pushes | latent | success | < 5 cm | final dist mean / median (cm) | pushes to success | off table | pred err (cm) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| ball/ablation_encoders/cross_attn_s0 | ball | 64 | 18.0 | 10 | mean | 25% | 34% | 10.7 / 7.1 | 4.4 | 8% | 12.55 |
| ball/ablation_encoders/cross_attn_s0 | ball | 64 | 18.0 | 10 | fitted | 50% | 59% | 10.0 / 2.0 | 3.2 | 3% | 11.09 |
| ball/ablation_encoders/cross_attn_s0 | ball | 64 | 18.0 | 10 | fitted_N | 61% | 67% | 9.1 / 1.7 | 3.7 | 5% | 9.74 |
| ball/ablation_encoders/cross_attn_s0 (hard) | ball | 64 | 20.1 | 2 | mean | 9% | 12% | 14.9 / 14.4 | 1.3 | 2% | 13.81 |
| ball/ablation_encoders/cross_attn_s0 (hard) | ball | 64 | 20.1 | 2 | fitted | 33% | 45% | 7.6 / 5.6 | 1.4 | 5% | 7.49 |
| ball/ablation_encoders/cross_attn_s0 (hard) | ball | 64 | 20.1 | 2 | fitted_N | 27% | 56% | 7.2 / 4.3 | 1.5 | 5% | 6.21 |
| ball/ablation_image_vs_pc/image_cls_s0 | ball | 64 | 18.0 | 10 | mean | 17% | 20% | 15.7 / 11.2 | 2.9 | 16% | 14.34 |
| ball/ablation_image_vs_pc/image_cls_s0 | ball | 64 | 18.0 | 10 | fitted | 8% | 16% | 28.4 / 20.0 | 2.8 | 23% | 19.99 |
| ball/ablation_image_vs_pc/image_cls_s0 | ball | 64 | 18.0 | 10 | fitted_N | 17% | 27% | 20.9 / 11.9 | 2.0 | 14% | 15.79 |
| ball/ablation_image_vs_pc/image_cls_s0 (hard) | ball | 64 | 20.1 | 2 | mean | 8% | 16% | 16.3 / 13.9 | 1.4 | 3% | 15.30 |
| ball/ablation_image_vs_pc/image_cls_s0 (hard) | ball | 64 | 20.1 | 2 | fitted | 6% | 16% | 13.5 / 11.1 | 1.2 | 2% | 11.52 |
| ball/ablation_image_vs_pc/image_cls_s0 (hard) | ball | 64 | 20.1 | 2 | fitted_N | 9% | 31% | 11.5 / 9.6 | 1.2 | 3% | 9.17 |
| ball/ablation_repr/image_af_cls_s0 | ball | 64 | 18.0 | 10 | mean | 27% | 36% | 14.8 / 8.1 | 5.7 | 9% | 14.23 |
| ball/ablation_repr/image_af_cls_s0 | ball | 64 | 18.0 | 10 | fitted | 31% | 41% | 12.9 / 5.5 | 3.6 | 11% | 16.00 |
| ball/ablation_repr/image_af_cls_s0 | ball | 64 | 18.0 | 10 | fitted_N | 25% | 36% | 15.4 / 5.8 | 5.1 | 16% | 16.14 |
| ball/ablation_repr/image_af_cls_s0 (hard) | ball | 64 | 20.1 | 2 | mean | 6% | 11% | 17.9 / 15.3 | 1.8 | 3% | 15.53 |
| ball/ablation_repr/image_af_cls_s0 (hard) | ball | 64 | 20.1 | 2 | fitted | 11% | 28% | 12.9 / 10.7 | 1.4 | 3% | 11.79 |
| ball/ablation_repr/image_af_cls_s0 (hard) | ball | 64 | 20.1 | 2 | fitted_N | 5% | 17% | 12.3 / 10.7 | 1.3 | 3% | 10.76 |
| ball/ablation_repr/image_af_cnn_s0 | ball | 64 | 18.0 | 10 | mean | 27% | 34% | 15.5 / 10.8 | 4.6 | 11% | 17.50 |
| ball/ablation_repr/image_af_cnn_s0 | ball | 64 | 18.0 | 10 | fitted | 34% | 39% | 15.1 / 8.6 | 4.6 | 3% | 13.91 |
| ball/ablation_repr/image_af_cnn_s0 | ball | 64 | 18.0 | 10 | fitted_N | 34% | 38% | 12.9 / 7.7 | 4.3 | 8% | 13.01 |
| ball/ablation_repr/image_af_cnn_s0 (hard) | ball | 64 | 20.1 | 2 | mean | 5% | 12% | 16.9 / 14.7 | 1.7 | 5% | 15.42 |
| ball/ablation_repr/image_af_cnn_s0 (hard) | ball | 64 | 20.1 | 2 | fitted | 6% | 28% | 11.7 / 10.3 | 1.2 | 0% | 11.13 |
| ball/ablation_repr/image_af_cnn_s0 (hard) | ball | 64 | 20.1 | 2 | fitted_N | 22% | 41% | 9.9 / 9.7 | 1.6 | 2% | 9.22 |
| ball/ensemble_wide | ball | 64 | 18.0 | 10 | mean | 27% | 39% | 13.0 / 6.4 | 4.7 | 3% | 13.86 |
| ball/ensemble_wide | ball | 64 | 18.0 | 10 | fitted | 70% | 77% | 5.6 / 1.6 | 3.5 | 2% | 7.38 |
| ball/ensemble_wide | ball | 64 | 18.0 | 10 | fitted_N | 75% | 86% | 3.5 / 1.5 | 3.4 | 0% | 4.91 |
| ball/ensemble_wide (hard) | ball | 64 | 20.1 | 2 | mean | 6% | 12% | 14.8 / 11.9 | 2.0 | 2% | 13.55 |
| ball/ensemble_wide (hard) | ball | 64 | 20.1 | 2 | fitted | 22% | 48% | 8.2 / 4.9 | 1.4 | 3% | 7.20 |
| ball/ensemble_wide (hard) | ball | 64 | 20.1 | 2 | fitted_N | 25% | 62% | 6.8 / 3.9 | 1.4 | 2% | 5.90 |
| ball | ball | 64 | 18.0 | 10 | mean | 38% | 45% | 10.7 / 4.6 | 6.0 | 9% | 11.48 |
| ball | ball | 64 | 18.0 | 10 | fitted | 52% | 66% | 7.7 / 2.0 | 3.2 | 6% | 8.32 |
| ball | ball | 64 | 18.0 | 10 | fitted_N | 47% | 61% | 10.3 / 1.9 | 4.4 | 9% | 10.12 |
| ball (hard) | ball | 64 | 20.1 | 2 | mean | 5% | 11% | 16.4 / 14.4 | 1.7 | 3% | 14.19 |
| ball (hard) | ball | 64 | 20.1 | 2 | fitted | 22% | 36% | 11.5 / 8.6 | 1.4 | 5% | 9.92 |
| ball (hard) | ball | 64 | 20.1 | 2 | fitted_N | 14% | 33% | 10.9 / 7.6 | 1.7 | 3% | 8.03 |
| block/ablation_arch/pointnet_wide_s0 | block | 64 | 16.3 | 10 | mean | 56% | 72% | 5.3 / 1.9 | 3.2 | 2% | 6.58 |
| block/ablation_arch/pointnet_wide_s0 | block | 64 | 16.3 | 10 | fitted | 78% | 83% | 3.0 / 1.6 | 2.3 | 2% | 4.86 |
| block/ablation_arch/pointnet_wide_s0 | block | 64 | 16.3 | 10 | fitted_N | 78% | 86% | 2.9 / 1.7 | 2.7 | 5% | 5.75 |
| block/ablation_arch/pointnet_wide_s0 (hard) | block | 64 | 20.1 | 2 | mean | 28% | 52% | 6.7 / 4.6 | 1.4 | 0% | 6.04 |
| block/ablation_arch/pointnet_wide_s0 (hard) | block | 64 | 20.1 | 2 | fitted | 36% | 62% | 5.1 / 3.3 | 1.3 | 0% | 4.46 |
| block/ablation_arch/pointnet_wide_s0 (hard) | block | 64 | 20.1 | 2 | fitted_N | 45% | 67% | 5.0 / 2.1 | 1.2 | 0% | 3.68 |
| block/ablation_image_vs_pc/image_cls_s0 | block | 64 | 16.3 | 10 | mean | 31% | 52% | 7.2 / 4.4 | 3.2 | 2% | 6.21 |
| block/ablation_image_vs_pc/image_cls_s0 | block | 64 | 16.3 | 10 | fitted | 27% | 50% | 8.5 / 4.9 | 3.5 | 2% | 6.84 |
| block/ablation_image_vs_pc/image_cls_s0 | block | 64 | 16.3 | 10 | fitted_N | 41% | 55% | 6.0 / 3.7 | 3.7 | 2% | 6.03 |
| block/ablation_image_vs_pc/image_cls_s0 (hard) | block | 64 | 20.1 | 2 | mean | 11% | 42% | 7.5 / 5.8 | 1.6 | 0% | 7.01 |
| block/ablation_image_vs_pc/image_cls_s0 (hard) | block | 64 | 20.1 | 2 | fitted | 14% | 42% | 8.0 / 5.5 | 1.2 | 3% | 6.92 |
| block/ablation_image_vs_pc/image_cls_s0 (hard) | block | 64 | 20.1 | 2 | fitted_N | 17% | 42% | 7.7 / 5.4 | 1.4 | 0% | 6.99 |
| block/ablation_repr/image_af_cls_s0 | block | 64 | 16.3 | 10 | mean | 59% | 73% | 4.5 / 1.9 | 4.2 | 0% | 7.50 |
| block/ablation_repr/image_af_cls_s0 | block | 64 | 16.3 | 10 | fitted | 50% | 64% | 4.9 / 2.0 | 4.0 | 3% | 7.25 |
| block/ablation_repr/image_af_cls_s0 | block | 64 | 16.3 | 10 | fitted_N | 64% | 78% | 4.6 / 1.6 | 4.5 | 2% | 7.00 |
| block/ablation_repr/image_af_cls_s0 (hard) | block | 64 | 20.1 | 2 | mean | 19% | 44% | 6.6 / 5.3 | 1.5 | 2% | 6.91 |
| block/ablation_repr/image_af_cls_s0 (hard) | block | 64 | 20.1 | 2 | fitted | 9% | 47% | 8.0 / 5.5 | 1.7 | 5% | 6.88 |
| block/ablation_repr/image_af_cls_s0 (hard) | block | 64 | 20.1 | 2 | fitted_N | 19% | 58% | 6.0 / 4.0 | 1.7 | 0% | 6.21 |
| block/ablation_repr/image_af_cnn_s0 | block | 64 | 16.3 | 10 | mean | 53% | 73% | 8.1 / 1.9 | 4.3 | 2% | 10.54 |
| block/ablation_repr/image_af_cnn_s0 | block | 64 | 16.3 | 10 | fitted | 66% | 78% | 4.1 / 1.8 | 4.0 | 2% | 7.07 |
| block/ablation_repr/image_af_cnn_s0 | block | 64 | 16.3 | 10 | fitted_N | 56% | 70% | 5.0 / 1.8 | 4.4 | 9% | 7.30 |
| block/ablation_repr/image_af_cnn_s0 (hard) | block | 64 | 20.1 | 2 | mean | 9% | 44% | 6.8 / 5.2 | 1.5 | 0% | 6.75 |
| block/ablation_repr/image_af_cnn_s0 (hard) | block | 64 | 20.1 | 2 | fitted | 19% | 53% | 5.8 / 4.2 | 1.8 | 0% | 6.25 |
| block/ablation_repr/image_af_cnn_s0 (hard) | block | 64 | 20.1 | 2 | fitted_N | 22% | 45% | 7.1 / 6.3 | 1.6 | 0% | 6.59 |
| block/ablation_repr/image_cnn_s0 | block | 64 | 16.3 | 10 | mean | 34% | 56% | 7.7 / 3.8 | 4.5 | 3% | 6.69 |
| block/ablation_repr/image_cnn_s0 | block | 64 | 16.3 | 10 | fitted | 38% | 55% | 7.9 / 2.9 | 3.7 | 6% | 8.04 |
| block/ablation_repr/image_cnn_s0 | block | 64 | 16.3 | 10 | fitted_N | 47% | 61% | 10.8 / 2.9 | 3.8 | 2% | 7.51 |
| block/ablation_repr/image_cnn_s0 (hard) | block | 64 | 20.1 | 2 | mean | 14% | 38% | 7.2 / 6.3 | 1.4 | 0% | 7.42 |
| block/ablation_repr/image_cnn_s0 (hard) | block | 64 | 20.1 | 2 | fitted | 11% | 36% | 7.7 / 6.6 | 1.3 | 0% | 7.89 |
| block/ablation_repr/image_cnn_s0 (hard) | block | 64 | 20.1 | 2 | fitted_N | 17% | 38% | 7.8 / 6.2 | 1.5 | 3% | 7.61 |
| block/ensemble_wide | block | 64 | 16.3 | 10 | mean | 66% | 75% | 3.8 / 1.6 | 3.1 | 3% | 6.37 |
| block/ensemble_wide | block | 64 | 16.3 | 10 | fitted | 83% | 91% | 2.4 / 1.5 | 2.1 | 0% | 4.95 |
| block/ensemble_wide | block | 64 | 16.3 | 10 | fitted_N | 88% | 89% | 3.2 / 1.6 | 2.4 | 3% | 4.09 |
| block/ensemble_wide (hard) | block | 64 | 20.1 | 2 | mean | 23% | 55% | 7.6 / 3.8 | 1.2 | 0% | 6.68 |
| block/ensemble_wide (hard) | block | 64 | 20.1 | 2 | fitted | 41% | 67% | 4.3 / 2.5 | 1.4 | 0% | 4.05 |
| block/ensemble_wide (hard) | block | 64 | 20.1 | 2 | fitted_N | 39% | 69% | 4.8 / 2.5 | 1.5 | 0% | 3.89 |
| block | block | 64 | 16.3 | 10 | mean | 66% | 78% | 5.0 / 1.8 | 3.6 | 2% | 6.83 |
| block | block | 64 | 16.3 | 10 | fitted | 80% | 86% | 4.1 / 1.4 | 3.4 | 2% | 8.36 |
| block | block | 64 | 16.3 | 10 | fitted_N | 83% | 88% | 4.4 / 1.4 | 2.8 | 2% | 7.61 |
| block (hard) | block | 64 | 20.1 | 2 | mean | 17% | 52% | 6.5 / 4.7 | 1.2 | 2% | 6.58 |
| block (hard) | block | 64 | 20.1 | 2 | fitted | 41% | 66% | 6.9 / 2.5 | 1.4 | 0% | 6.89 |
| block (hard) | block | 64 | 20.1 | 2 | fitted_N | 36% | 56% | 8.2 / 3.2 | 1.3 | 3% | 6.18 |
| block_big/ensemble_wide | block | 64 | 16.3 | 10 | mean | 67% | 78% | 3.8 / 1.7 | 3.0 | 5% | 6.99 |
| block_big/ensemble_wide | block | 64 | 16.3 | 10 | fitted | 80% | 88% | 2.3 / 1.5 | 2.4 | 2% | 4.13 |
| block_big/ensemble_wide | block | 64 | 16.3 | 10 | fitted_N | 81% | 88% | 3.8 / 1.6 | 2.1 | 2% | 4.96 |
| block_big/ensemble_wide (hard) | block | 64 | 20.1 | 2 | mean | 27% | 56% | 6.8 / 3.9 | 1.3 | 2% | 6.24 |
| block_big/ensemble_wide (hard) | block | 64 | 20.1 | 2 | fitted | 42% | 72% | 4.0 / 2.4 | 1.4 | 0% | 3.32 |
| block_big/ensemble_wide (hard) | block | 64 | 20.1 | 2 | fitted_N | 45% | 70% | 4.1 / 2.3 | 1.4 | 3% | 3.63 |
| block_big | block | 64 | 16.3 | 10 | mean | 61% | 70% | 7.8 / 1.9 | 3.2 | 3% | 43.93 |
| block_big | block | 64 | 16.3 | 10 | fitted | 73% | 81% | 3.7 / 1.7 | 3.3 | 2% | 5.93 |
| block_big | block | 64 | 16.3 | 10 | fitted_N | 69% | 77% | 4.6 / 1.7 | 3.0 | 3% | 6.88 |
| block_big (hard) | block | 64 | 20.1 | 2 | mean | 28% | 58% | 6.3 / 4.1 | 1.2 | 2% | 6.78 |
| block_big (hard) | block | 64 | 20.1 | 2 | fitted | 30% | 59% | 5.2 / 3.4 | 1.3 | 3% | 4.94 |
| block_big (hard) | block | 64 | 20.1 | 2 | fitted_N | 42% | 70% | 4.1 / 2.4 | 1.1 | 2% | 3.90 |

### ball/ablation_arch

task ball, 3478 transitions (train 3120, val 358), no-motion baseline 8.13 cm (moved: 15.35 cm), seeds [0, 1]

Errors in cm (mean point-flow error on the val split); `moved` = pushes that moved the object > 5 mm; `rel` = median error / distance moved; the last columns bin by distance moved.

| variant | n_train | steps | params | val_err_cm | val_err_moved_cm | val_rel_err_moved | err_0.5-5cm | err_5.0-15cm | err_15.0-1000cm | min |
|---|---|---|---|---|---|---|---|---|---|---|
| pointnet | 3120 | 5750 | 128k | 4.31 ± 0.10 | 7.84 ± 0.20 | 0.36 ± 0.04 | 1.96 ± 0.18 | 4.28 ± 0.26 | 14.83 ± 0.19 | 0.6 |
| pointnet_wide | 3120 | 4250 | 707k | 4.47 ± 0.01 | 8.20 ± 0.06 | 0.41 ± 0.03 | 2.18 ± 0.07 | 4.85 ± 0.50 | 15.19 ± 0.21 | 0.6 |
| pointnet_res | 3120 | 5100 | 591k | 4.40 ± 0.02 | 7.98 ± 0.06 | 0.38 ± 0.02 | 2.00 ± 0.05 | 4.37 ± 0.23 | 15.08 ± 0.02 | 0.7 |
| pointnet_res_wide | 3120 | 6300 | 2288k | 4.28 ± 0.03 | 7.72 ± 0.01 | 0.40 ± 0.03 | 1.95 ± 0.03 | 4.60 ± 0.01 | 14.38 ± 0.05 | 1.0 |
| cross_attn_res | 3120 | 6850 | 708k | 4.38 ± 0.01 | 8.02 ± 0.00 | 0.42 ± 0.00 | 1.98 ± 0.01 | 4.59 ± 0.10 | 15.07 ± 0.06 | 1.8 |

### ball/ablation_data

task ball, 3478 transitions (train 3120, val 358), no-motion baseline 8.13 cm (moved: 15.35 cm), seeds [0, 1]

Errors in cm (mean point-flow error on the val split); `moved` = pushes that moved the object > 5 mm; `rel` = median error / distance moved; the last columns bin by distance moved.

| variant | n_train | steps | params | val_err_cm | val_err_moved_cm | val_rel_err_moved | err_0.5-5cm | err_5.0-15cm | err_15.0-1000cm | min |
|---|---|---|---|---|---|---|---|---|---|---|
| pointnet | 3120 | 6450 | 128k | 4.34 ± 0.01 | 7.91 ± 0.02 | 0.37 ± 0.01 | 1.92 ± 0.08 | 4.94 ± 0.51 | 14.65 ± 0.17 | 0.8 |
| data12 | 390 | 3300 | 128k | 6.20 ± 0.28 | 11.44 ± 0.47 | 0.64 ± 0.02 | 3.59 ± 0.86 | 6.79 ± 0.15 | 20.73 ± 0.34 | 0.4 |
| data25 | 780 | 4000 | 128k | 5.15 ± 0.40 | 9.31 ± 0.73 | 0.51 ± 0.10 | 3.10 ± 0.97 | 5.22 ± 0.61 | 16.90 ± 0.60 | 0.4 |
| data50 | 1560 | 5400 | 128k | 4.71 ± 0.11 | 8.51 ± 0.14 | 0.41 ± 0.01 | 2.42 ± 0.16 | 4.75 ± 0.20 | 15.79 ± 0.09 | 0.6 |

### ball/ablation_encoders

task ball, 3478 transitions (train 3120, val 358), no-motion baseline 8.13 cm (moved: 15.35 cm), seeds [0, 1]

Errors in cm (mean point-flow error on the val split); `moved` = pushes that moved the object > 5 mm; `rel` = median error / distance moved; the last columns bin by distance moved.

| variant | n_train | steps | params | val_err_cm | val_err_moved_cm | val_rel_err_moved | err_0.5-5cm | err_5.0-15cm | err_15.0-1000cm | min |
|---|---|---|---|---|---|---|---|---|---|---|
| pointnet | 3120 | 5750 | 128k | 4.31 ± 0.10 | 7.84 ± 0.20 | 0.36 ± 0.04 | 1.96 ± 0.18 | 4.28 ± 0.26 | 14.83 ± 0.19 | 0.7 |
| pointnet_early | 3120 | 5300 | 129k | 4.42 ± 0.02 | 8.04 ± 0.03 | 0.40 ± 0.02 | 2.11 ± 0.12 | 4.22 ± 0.04 | 15.23 ± 0.06 | 0.6 |
| cross_attn | 3120 | 5250 | 245k | 4.26 ± 0.06 | 7.81 ± 0.00 | 0.34 ± 0.01 | 1.88 ± 0.16 | 4.41 ± 0.05 | 14.75 ± 0.10 | 1.3 |
| transformer | 3120 | 5350 | 502k | 4.29 ± 0.01 | 7.87 ± 0.00 | 0.32 ± 0.03 | 1.91 ± 0.12 | 4.66 ± 0.08 | 14.74 ± 0.15 | 2.6 |

### ball/ablation_image_vs_pc

task ball, 3478 transitions (train 3120, val 358), no-motion baseline 8.13 cm (moved: 15.35 cm), seeds [0, 1]

Errors in cm (mean point-flow error on the val split); `moved` = pushes that moved the object > 5 mm; `rel` = median error / distance moved; the last columns bin by distance moved.

| variant | n_train | steps | params | val_err_cm | val_err_moved_cm | val_rel_err_moved | err_0.5-5cm | err_5.0-15cm | err_15.0-1000cm | min |
|---|---|---|---|---|---|---|---|---|---|---|
| pointnet | 3120 | 6150 | 128k | 4.34 ± 0.05 | 7.92 ± 0.04 | 0.39 ± 0.00 | 1.91 ± 0.02 | 4.79 ± 0.38 | 14.77 ± 0.10 | 0.6 |
| image_cls | 3120 | 5050 | 334k | 5.24 ± 0.07 | 8.94 ± 0.21 | 0.49 ± 0.07 | 2.12 ± 0.13 | 5.13 ± 0.08 | 16.87 ± 0.46 | 0.3 |
| image_attn | 3120 | 5500 | 271k | 5.18 ± 0.11 | 8.97 ± 0.14 | 0.51 ± 0.00 | 2.30 ± 0.17 | 5.26 ± 0.12 | 16.71 ± 0.56 | 0.6 |
| no_mirror | 3120 | 4500 | 128k | 4.53 ± 0.04 | 8.27 ± 0.09 | 0.38 ± 0.00 | 2.24 ± 0.21 | 4.48 ± 0.14 | 15.54 ± 0.13 | 0.4 |

### ball/ablation_latent_training

task ball, 3478 transitions (train 3120, val 358), no-motion baseline 8.13 cm (moved: 15.35 cm), seeds [0, 1]

Errors in cm (mean point-flow error on the val split); `moved` = pushes that moved the object > 5 mm; `rel` = median error / distance moved; the last columns bin by distance moved.

| variant | n_train | steps | params | val_err_cm | val_err_moved_cm | val_rel_err_moved | err_0.5-5cm | err_5.0-15cm | err_15.0-1000cm | min |
|---|---|---|---|---|---|---|---|---|---|---|
| pointnet | 3120 | 5750 | 128k | 4.31 ± 0.10 | 7.84 ± 0.20 | 0.36 ± 0.04 | 1.96 ± 0.18 | 4.28 ± 0.26 | 14.83 ± 0.19 | 0.7 |
| latent0 | 3120 | 5900 | 126k | 6.60 ± 0.05 | 11.83 ± 0.10 | 0.74 ± 0.00 | 4.55 ± 0.06 | 6.11 ± 0.01 | 21.28 ± 0.29 | 0.7 |
| latent16 | 3120 | 5700 | 130k | 4.40 ± 0.05 | 8.01 ± 0.04 | 0.37 ± 0.00 | 2.02 ± 0.26 | 4.38 ± 0.26 | 15.14 ± 0.28 | 0.7 |
| relative_loss | 3120 | 6050 | 128k | 5.24 ± 0.29 | 9.78 ± 0.54 | 0.55 ± 0.07 | 1.55 ± 0.04 | 5.56 ± 0.08 | 19.13 ± 1.23 | 0.7 |
| no_noise | 3120 | 5700 | 128k | 4.34 ± 0.00 | 7.95 ± 0.05 | 0.41 ± 0.01 | 1.84 ± 0.10 | 4.41 ± 0.23 | 15.12 ± 0.16 | 0.6 |
| lr3e-4 | 3120 | 11800 | 128k | 4.44 ± 0.00 | 8.07 ± 0.01 | 0.38 ± 0.02 | 2.01 ± 0.16 | 4.43 ± 0.08 | 15.28 ± 0.06 | 1.4 |

### ball/ablation_repr

task ball, 3478 transitions (train 3120, val 358), no-motion baseline 8.13 cm (moved: 15.35 cm), seeds [0, 1]

Errors in cm (mean point-flow error on the val split); `moved` = pushes that moved the object > 5 mm; `rel` = median error / distance moved; the last columns bin by distance moved.

| variant | n_train | steps | params | val_err_cm | val_err_moved_cm | val_rel_err_moved | err_0.5-5cm | err_5.0-15cm | err_15.0-1000cm | min |
|---|---|---|---|---|---|---|---|---|---|---|
| pointnet | 3120 | 5750 | 128k | 4.31 ± 0.10 | 7.84 ± 0.20 | 0.36 ± 0.04 | 1.96 ± 0.18 | 4.28 ± 0.26 | 14.83 ± 0.19 | 0.6 |
| image_cls | 3120 | 4700 | 334k | 5.24 ± 0.07 | 9.15 ± 0.08 | 0.54 ± 0.02 | 2.26 ± 0.06 | 5.42 ± 0.10 | 17.10 ± 0.19 | 0.3 |
| pointnet_geom | 3120 | 5000 | 128k | 4.28 ± 0.03 | 7.81 ± 0.05 | 0.36 ± 0.03 | 1.91 ± 0.11 | 4.65 ± 0.17 | 14.60 ± 0.08 | 0.5 |
| pointnet_objcentric | 3120 | 4300 | 129k | 4.26 ± 0.00 | 7.77 ± 0.01 | 0.40 ± 0.01 | 2.05 ± 0.05 | 4.59 ± 0.19 | 14.42 ± 0.17 | 0.5 |
| pointnet_dino | 3120 | 4250 | 392k | 4.55 ± 0.08 | 8.20 ± 0.10 | 0.37 ± 0.02 | 2.37 ± 0.17 | 4.68 ± 0.30 | 15.13 ± 0.27 | 0.5 |
| image_af_cls | 3120 | 4550 | 333k | 5.17 ± 0.06 | 9.12 ± 0.01 | 0.54 ± 0.02 | 2.64 ± 0.06 | 5.55 ± 0.31 | 16.61 ± 0.17 | 0.3 |
| image_cnn | 3120 | 6350 | 559k | 6.37 ± 0.03 | 10.70 ± 0.01 | 0.71 ± 0.05 | 3.14 ± 0.42 | 6.35 ± 0.43 | 19.58 ± 0.12 | 1.9 |
| image_af_cnn | 3120 | 4000 | 559k | 5.02 ± 0.04 | 9.00 ± 0.15 | 0.45 ± 0.00 | 2.71 ± 0.33 | 4.73 ± 0.06 | 16.77 ± 0.05 | 1.2 |

### block/ablation_arch

task block, 3734 transitions (train 3342, val 392), no-motion baseline 5.80 cm (moved: 11.60 cm), seeds [0, 1]

Errors in cm (mean point-flow error on the val split); `moved` = pushes that moved the object > 5 mm; `rel` = median error / distance moved; the last columns bin by distance moved.

| variant | n_train | steps | params | val_err_cm | val_err_moved_cm | val_rel_err_moved | err_0.5-5cm | err_5.0-15cm | err_15.0-1000cm | min |
|---|---|---|---|---|---|---|---|---|---|---|
| pointnet | 3342 | 6900 | 128k | 2.46 ± 0.00 | 4.43 ± 0.01 | 0.28 ± 0.01 | 2.14 ± 0.08 | 4.01 ± 0.05 | 7.00 ± 0.05 | 0.7 |
| pointnet_wide | 3342 | 6150 | 707k | 2.37 ± 0.05 | 4.23 ± 0.02 | 0.29 ± 0.01 | 2.08 ± 0.05 | 3.79 ± 0.09 | 6.68 ± 0.10 | 0.8 |
| pointnet_res | 3342 | 8350 | 591k | 2.50 ± 0.01 | 4.41 ± 0.07 | 0.33 ± 0.04 | 2.31 ± 0.27 | 3.95 ± 0.22 | 6.85 ± 0.21 | 1.1 |
| pointnet_res_wide | 3342 | 6700 | 2288k | 2.43 ± 0.01 | 4.33 ± 0.03 | 0.26 ± 0.00 | 2.23 ± 0.04 | 3.59 ± 0.07 | 7.06 ± 0.01 | 0.9 |
| cross_attn_res | 3342 | 13050 | 708k | 2.39 ± 0.01 | 4.32 ± 0.01 | 0.26 ± 0.02 | 2.19 ± 0.07 | 3.56 ± 0.15 | 7.12 ± 0.07 | 2.5 |

### block/ablation_data

task block, 3734 transitions (train 3342, val 392), no-motion baseline 5.80 cm (moved: 11.60 cm), seeds [0, 1]

Errors in cm (mean point-flow error on the val split); `moved` = pushes that moved the object > 5 mm; `rel` = median error / distance moved; the last columns bin by distance moved.

| variant | n_train | steps | params | val_err_cm | val_err_moved_cm | val_rel_err_moved | err_0.5-5cm | err_5.0-15cm | err_15.0-1000cm | min |
|---|---|---|---|---|---|---|---|---|---|---|
| pointnet | 3342 | 6900 | 128k | 2.43 ± 0.00 | 4.36 ± 0.01 | 0.29 ± 0.02 | 1.87 ± 0.13 | 3.99 ± 0.02 | 7.05 ± 0.14 | 0.8 |
| data12 | 417 | 3450 | 128k | 3.70 ± 0.02 | 6.80 ± 0.12 | 0.70 ± 0.03 | 2.98 ± 0.22 | 5.92 ± 0.05 | 11.28 ± 0.62 | 0.4 |
| data25 | 835 | 10200 | 128k | 3.31 ± 0.19 | 5.68 ± 0.23 | 0.45 ± 0.04 | 2.92 ± 0.18 | 4.96 ± 0.31 | 8.98 ± 0.19 | 1.1 |
| data50 | 1671 | 6100 | 128k | 2.78 ± 0.01 | 4.91 ± 0.08 | 0.38 ± 0.02 | 2.45 ± 0.07 | 4.05 ± 0.02 | 8.13 ± 0.16 | 0.7 |

### block/ablation_encoders

task block, 3734 transitions (train 3342, val 392), no-motion baseline 5.80 cm (moved: 11.60 cm), seeds [0, 1]

Errors in cm (mean point-flow error on the val split); `moved` = pushes that moved the object > 5 mm; `rel` = median error / distance moved; the last columns bin by distance moved.

| variant | n_train | steps | params | val_err_cm | val_err_moved_cm | val_rel_err_moved | err_0.5-5cm | err_5.0-15cm | err_15.0-1000cm | min |
|---|---|---|---|---|---|---|---|---|---|---|
| pointnet | 3342 | 6900 | 128k | 2.46 ± 0.00 | 4.43 ± 0.01 | 0.28 ± 0.01 | 2.14 ± 0.08 | 4.01 ± 0.05 | 7.00 ± 0.05 | 0.7 |
| pointnet_early | 3342 | 9900 | 129k | 2.40 ± 0.01 | 4.29 ± 0.12 | 0.32 ± 0.04 | 1.88 ± 0.06 | 4.01 ± 0.19 | 6.82 ± 0.20 | 1.2 |
| cross_attn | 3342 | 9800 | 245k | 2.61 ± 0.09 | 4.60 ± 0.20 | 0.34 ± 0.02 | 2.30 ± 0.07 | 3.96 ± 0.09 | 7.41 ± 0.44 | 1.9 |
| transformer | 3342 | 6100 | 502k | 2.63 ± 0.03 | 4.74 ± 0.05 | 0.33 ± 0.01 | 2.12 ± 0.08 | 4.74 ± 0.03 | 7.16 ± 0.11 | 2.0 |

### block/ablation_image_vs_pc

task block, 3734 transitions (train 3342, val 392), no-motion baseline 5.80 cm (moved: 11.60 cm), seeds [0, 1]

Errors in cm (mean point-flow error on the val split); `moved` = pushes that moved the object > 5 mm; `rel` = median error / distance moved; the last columns bin by distance moved.

| variant | n_train | steps | params | val_err_cm | val_err_moved_cm | val_rel_err_moved | err_0.5-5cm | err_5.0-15cm | err_15.0-1000cm | min |
|---|---|---|---|---|---|---|---|---|---|---|
| pointnet | 3342 | 8750 | 128k | 2.46 ± 0.02 | 4.40 ± 0.07 | 0.29 ± 0.01 | 2.14 ± 0.12 | 3.81 ± 0.19 | 7.13 ± 0.12 | 2.0 |
| image_cls | 3342 | 4100 | 334k | 2.97 ± 0.03 | 5.30 ± 0.05 | 0.43 ± 0.00 | 2.23 ± 0.06 | 5.25 ± 0.05 | 8.20 ± 0.04 | 0.4 |
| image_attn | 3342 | 5700 | 271k | 3.10 ± 0.03 | 5.63 ± 0.14 | 0.53 ± 0.00 | 2.50 ± 0.00 | 5.59 ± 0.15 | 8.57 ± 0.58 | 0.9 |
| no_mirror | 3342 | 6450 | 128k | 2.70 ± 0.01 | 4.91 ± 0.01 | 0.36 ± 0.02 | 2.39 ± 0.07 | 4.35 ± 0.05 | 7.84 ± 0.06 | 0.7 |

### block/ablation_latent_training

task block, 3734 transitions (train 3342, val 392), no-motion baseline 5.80 cm (moved: 11.60 cm), seeds [0, 1]

Errors in cm (mean point-flow error on the val split); `moved` = pushes that moved the object > 5 mm; `rel` = median error / distance moved; the last columns bin by distance moved.

| variant | n_train | steps | params | val_err_cm | val_err_moved_cm | val_rel_err_moved | err_0.5-5cm | err_5.0-15cm | err_15.0-1000cm | min |
|---|---|---|---|---|---|---|---|---|---|---|
| pointnet | 3342 | 6900 | 128k | 2.46 ± 0.00 | 4.43 ± 0.01 | 0.28 ± 0.01 | 2.14 ± 0.08 | 4.01 ± 0.05 | 7.00 ± 0.05 | 0.7 |
| latent0 | 3342 | 6350 | 126k | 3.71 ± 0.03 | 6.18 ± 0.06 | 0.55 ± 0.00 | 3.89 ± 0.54 | 5.03 ± 0.02 | 9.52 ± 0.64 | 0.7 |
| latent16 | 3342 | 7600 | 130k | 2.55 ± 0.00 | 4.53 ± 0.00 | 0.30 ± 0.01 | 2.20 ± 0.09 | 3.92 ± 0.04 | 7.34 ± 0.13 | 0.9 |
| relative_loss | 3342 | 6250 | 128k | 3.17 ± 0.18 | 6.27 ± 0.37 | 0.60 ± 0.08 | 2.30 ± 0.16 | 5.19 ± 0.00 | 11.10 ± 0.97 | 0.7 |
| no_noise | 3342 | 12100 | 128k | 2.43 ± 0.03 | 4.41 ± 0.01 | 0.30 ± 0.02 | 2.21 ± 0.10 | 3.76 ± 0.12 | 7.14 ± 0.18 | 1.1 |
| lr3e-4 | 3342 | 9850 | 128k | 2.60 ± 0.04 | 4.73 ± 0.10 | 0.34 ± 0.01 | 2.31 ± 0.11 | 4.18 ± 0.16 | 7.56 ± 0.03 | 0.9 |

### block/ablation_repr

task block, 3734 transitions (train 3342, val 392), no-motion baseline 5.80 cm (moved: 11.60 cm), seeds [0, 1]

Errors in cm (mean point-flow error on the val split); `moved` = pushes that moved the object > 5 mm; `rel` = median error / distance moved; the last columns bin by distance moved.

| variant | n_train | steps | params | val_err_cm | val_err_moved_cm | val_rel_err_moved | err_0.5-5cm | err_5.0-15cm | err_15.0-1000cm | min |
|---|---|---|---|---|---|---|---|---|---|---|
| pointnet | 3342 | 8750 | 128k | 2.46 ± 0.02 | 4.40 ± 0.07 | 0.29 ± 0.01 | 2.14 ± 0.12 | 3.81 ± 0.19 | 7.13 ± 0.12 | 1.0 |
| image_cls | 3342 | 4250 | 334k | 2.95 ± 0.10 | 5.23 ± 0.13 | 0.42 ± 0.03 | 2.23 ± 0.01 | 5.28 ± 0.25 | 7.96 ± 0.13 | 0.4 |
| pointnet_geom | 3342 | 6950 | 128k | 2.42 ± 0.06 | 4.40 ± 0.09 | 0.29 ± 0.02 | 1.92 ± 0.09 | 4.01 ± 0.36 | 7.10 ± 0.20 | 0.8 |
| pointnet_objcentric | 3342 | 6300 | 129k | 2.58 ± 0.02 | 4.59 ± 0.07 | 0.34 ± 0.01 | 1.92 ± 0.01 | 4.37 ± 0.03 | 7.29 ± 0.18 | 0.7 |
| pointnet_dino | 3342 | 6400 | 392k | 2.54 ± 0.06 | 4.58 ± 0.14 | 0.30 ± 0.00 | 2.19 ± 0.15 | 4.13 ± 0.07 | 7.26 ± 0.19 | 0.8 |
| image_af_cls | 3342 | 3750 | 333k | 3.07 ± 0.02 | 5.34 ± 0.02 | 0.46 ± 0.01 | 2.30 ± 0.04 | 5.20 ± 0.08 | 8.29 ± 0.01 | 0.3 |
| image_cnn | 3342 | 4400 | 559k | 3.38 ± 0.03 | 5.97 ± 0.01 | 0.51 ± 0.01 | 2.68 ± 0.08 | 5.37 ± 0.02 | 9.65 ± 0.01 | 1.3 |
| image_af_cnn | 3342 | 5450 | 559k | 2.93 ± 0.06 | 5.22 ± 0.12 | 0.41 ± 0.01 | 2.29 ± 0.03 | 4.53 ± 0.05 | 8.67 ± 0.34 | 1.5 |

### block_big/ablation_big

task block, 8642 transitions (train 7763, val 879), no-motion baseline 5.09 cm (moved: 10.97 cm), seeds [0, 1]

Errors in cm (mean point-flow error on the val split); `moved` = pushes that moved the object > 5 mm; `rel` = median error / distance moved; the last columns bin by distance moved.

| variant | n_train | steps | params | val_err_cm | val_err_moved_cm | val_rel_err_moved | err_0.5-5cm | err_5.0-15cm | err_15.0-1000cm | min |
|---|---|---|---|---|---|---|---|---|---|---|
| pointnet | 7763 | 12200 | 128k | 2.12 ± 0.04 | 4.07 ± 0.08 | 0.29 ± 0.01 | 1.80 ± 0.08 | 2.98 ± 0.10 | 8.29 ± 0.05 | 1.7 |
| cross_attn | 7763 | 12200 | 245k | 2.06 ± 0.04 | 3.92 ± 0.07 | 0.26 ± 0.00 | 1.67 ± 0.02 | 2.80 ± 0.03 | 8.16 ± 0.18 | 3.0 |
| image_cls | 7763 | 6350 | 334k | 2.45 ± 0.00 | 4.69 ± 0.02 | 0.42 ± 0.02 | 1.69 ± 0.08 | 4.09 ± 0.04 | 8.99 ± 0.24 | 0.6 |
| image_attn | 7763 | 5850 | 271k | 2.47 ± 0.00 | 4.87 ± 0.03 | 0.46 ± 0.00 | 1.92 ± 0.06 | 4.25 ± 0.07 | 9.14 ± 0.13 | 1.0 |
| pointnet_res_wide | 7763 | 10800 | 2288k | 2.06 ± 0.04 | 3.92 ± 0.04 | 0.24 ± 0.01 | 1.55 ± 0.01 | 2.75 ± 0.04 | 8.37 ± 0.23 | 1.8 |
| data12 | 970 | 3900 | 128k | 3.04 ± 0.06 | 6.02 ± 0.05 | 0.59 ± 0.01 | 2.25 ± 0.03 | 4.98 ± 0.14 | 11.84 ± 0.06 | 1.2 |
| data25 | 1940 | 5250 | 128k | 2.66 ± 0.06 | 5.15 ± 0.03 | 0.43 ± 0.01 | 1.92 ± 0.03 | 4.16 ± 0.08 | 10.29 ± 0.27 | 1.4 |
| data50 | 3881 | 8050 | 128k | 2.39 ± 0.04 | 4.52 ± 0.06 | 0.33 ± 0.00 | 1.83 ± 0.03 | 3.53 ± 0.05 | 9.08 ± 0.27 | 2.2 |
