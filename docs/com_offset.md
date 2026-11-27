# Blocks with an off-centre centre of mass

Can a fitted physics latent handle a block whose weight is not in the middle? We moved the block's
centre of mass (COM) up to 1.5 cm off centre, inside a 4 cm cube, and reran the pipeline: training,
latent fitting, and MPC on unseen blocks.

Short answer: fitting the latent still helps the planner reach goals, but the latent never learns
where the weight is, and rotation predictions don't get better after fitting. A cube looks the same
after a 90 degree turn, so one latent per block can't say which side is heavy. That is the case for
a per-point latent map, which we haven't built.

## What changed in the code

Defaults are unchanged. With `com_offset = 0` the physics pool, the RNG stream and the checkpoints
are the same as before.

- `env/push_block/env.py`: `BlockPhysicsConfig` has `com_x`, `com_y` (m, block frame). The COM is set
  with `set_mass_and_inertia`; the inertia stays that of a uniform cube.
- `train_dynamics.py`: `--com-offset R` samples each block's COM uniformly in a disk of radius R.
  The validation metrics now include rotation error.
- `eval_mpc.py`: `--com-offset` for the test blocks. It defaults to the checkpoint's training value,
  so pass it explicitly for an ensemble directory.
- `eval_latent_inference.py`: reports rotation error, and a leave-one-out linear R^2 from the fitted
  latent to log mass, friction and the COM.
- `scripts/compare_encoders.py`, `scripts/train_ensemble.py`: `--no-mirror-aug`, plus rotation error.
  Mirror augmentation flips the block, which is wrong once the COM is off centre, so every run
  here has it off, the uniform baseline included.
- `env/demo_com_offset.py`: the sanity check below.
- `modal_app.py`, `scripts/com_experiments.sh`, `scripts/collect_com_results.py`: how the runs were
  launched and tabulated.

## Running it

Everything that touches the simulator needs a Linux GPU with Vulkan. We ran it on Modal:

```
modal run --detach modal_app.py --name train_uniform --cmd "python train_dynamics.py --run-dir runs/uniform_nomirror --no-mirror-aug"
modal run --detach modal_app.py --name train_com015 --cmd "python train_dynamics.py --run-dir runs/com015 --com-offset 0.015 --no-mirror-aug"
modal run --detach modal_app.py --name offline --cmd "bash scripts/com_experiments.sh offline"
modal run --detach modal_app.py --name eval --cmd "bash scripts/com_experiments.sh eval"
modal run --detach modal_app.py --name eval_ensemble --cmd "bash scripts/com_experiments.sh eval_ensemble"
```

Outputs land on the Modal volume `latent-physics-runs` (mounted as `runs/`), logs in
`runs/logs/<name>.log`. On a local Linux GPU, run the same commands without the `modal run` wrapper.
Fetch the JSON results with `modal volume get` and run `python scripts/collect_com_results.py --runs runs`
for the full tables. All of it took about 6 GPU hours, mostly on L4s.

Modal note: a running container doesn't see files that other jobs commit to the volume, so start
jobs that depend on each other one after the other.

## Sanity check

`python -m env.demo_com_offset --video com_sanity.mp4` pushes six blocks straight through the middle
of a face. A uniform block turns less than 1 degree per push. With the COM 1.5 cm to one side it turns
14 to 20 degrees per push, and the opposite offset turns the other way. A 1.8 cm offset turns no more
than 1.5 cm. Video: [`demos/com_offset/com_sanity.mp4`](../demos/com_offset/com_sanity.mp4).

## Results

Two models with the same settings and seed: `uniform_nomirror` (trained on uniform blocks) and
`com015` (COM up to 1.5 cm). Tests use 64 unseen physics settings (test seed 1000).

### Goal reaching with MPC

Success means the block ends within 2 cm of the goal (start-to-goal distance 16.3 cm, up to 10 pushes).
"fitted" fits the latent on 4 probe pushes, "fitted_N" on 16.

| model | test blocks | mean latent | fitted | fitted_N |
|---|---|---|---|---|
| uniform_nomirror | uniform | 56% | 66% | 70% |
| uniform_nomirror | COM 1.5 cm | 66% | 72% | 71% |
| com015 | COM 1.5 cm | 56% | 64% | 65% |
| com015, 5-model ensemble | COM 1.5 cm | 61% | 78% | 82% |

Each cell averages 2 or 3 repeats. The GPU sim isn't deterministic and the same eval has given 59%
and 73%, so treat gaps under 10 points as noise. `collect_com_results.py` also prints final distances
and the MPC's prediction errors for each run.

- Fitting beats the mean latent for every model on COM blocks, by 6 to 21 points.
- The ensemble benefits most. With 16 probes its final distance drops from 6.0 to 2.8 cm and its
  in-episode prediction error from 7.6 to 4.1 cm.
- A single model trained on COM blocks did no better on them than the uniform model. If the latent
  can't encode the COM, varying it in training mostly adds noise.

### One-push prediction error

Latent fitted on the first 8 pushes in each environment, error measured on the last 6. "Oracle" fits
on all 16 pushes, held-out ones included, so it is the best a single global latent can do.

| model / blocks | latent | position (cm) | rotation (deg) | rotation, pushes that moved it (deg) |
|---|---|---|---|---|
| uniform_nomirror / uniform | mean | 3.20 | 12.8 | 18.3 |
| uniform_nomirror / uniform | fitted | 2.39 | 12.5 | 20.0 |
| uniform_nomirror / uniform | oracle | 1.18 | 9.2 | 15.3 |
| com015 / COM | mean | 4.11 | 17.1 | 24.6 |
| com015 / COM | fitted | 3.36 | 17.3 | 25.8 |
| com015 / COM | oracle | 1.84 | 14.9 | 23.8 |

Fitting fixes position on both kinds of block. Rotation barely moves, and on COM blocks even the
oracle stays near 15 degrees.

### Encoders and ensemble (offline, on the com015 dataset)

3716 transitions, same validation split for every model.

| model | val position (cm) | val rotation (deg) |
|---|---|---|
| pointnet | 2.32 | 13.2 |
| pointnet_early | 2.24 | 13.6 |
| cross_attn | 2.18 | 13.2 |
| transformer | 2.30 | 14.8 |
| ensemble of 5 cross_attn | 2.08 | 13.1 |

`cross_attn` is the best single encoder, by a small margin. Adding ensemble members lowers position
error a little and leaves rotation where it was, so the rotation error isn't model variance.

### What the latent encodes

Leave-one-out linear R^2 from the oracle latent to the true physics, over 61 test environments.
0 means no better than guessing the mean.

| model | log mass | mu_s | mu_d | com_x | com_y | COM distance |
|---|---|---|---|---|---|---|
| uniform_nomirror | 0.68 | 0.09 | 0.10 | | | |
| com015 | 0.65 | -0.06 | -0.07 | -0.32 | -0.21 | -0.23 |

Mass is in the latent; friction mostly isn't, in either model. The COM isn't there at all, not even
its distance from the centre, which doesn't depend on which way the block faces.

## Caveats

- One training seed per model, and 2 or 3 MPC repeats per row.
- The COM model's validation error during training (2.73 cm, 15.5 degrees) is worse than the offline
  encoder runs on the same data, which train longer on the final dataset.
- Averages over all pushes include pushes that miss the block, where the true rotation is about 0.
  The "pushes that moved it" column is the fairer rotation number.

## Videos

- [`demos/com_offset/mpc_com015_ensemble_fitted.mp4`](../demos/com_offset/mpc_com015_ensemble_fitted.mp4):
  the ensemble with a fitted latent pushing COM blocks to goals. Each tile shows the block's COM.
- [`demos/com_offset/com_sanity.mp4`](../demos/com_offset/com_sanity.mp4): the sanity check above.
