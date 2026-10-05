# lewm-cus

A small reimplementation of **LeWM** (LeWorldModel), a joint-embedding predictive
world model, trained from pixels on four MMBench2 DMControl tasks:
**acrobot-swingup**, **cartpole-swingup**, **pendulum-swingup** and **reacher-easy**.
Frozen **DINOv2** features (DINO-WM) and the true state are the comparisons.

The world model predicts the next latent from the last few latents and the
actions taken; it never decodes back to pixels. With a trainable encoder,
SIGReg keeps the embedding Gaussian so it cannot collapse. The analyses then
ask what the latent holds (a linear probe for the joint angles, a pixel decoder)
and how prediction error compounds over autoregressive rollouts.

## Layout

```
run.py                  one task end to end: data -> train -> probe -> decoder -> rollout
config.py               every setting, as dataclasses + presets; data.task picks the task
train.py                train the world model
datasets/
  inspect_mmbench.py    stats and sample frames of a raw MMBench download
  prepare_mmbench.py    raw download -> the HDF5 table train.py reads
ablations/
  common.py             shared pieces: model loading, episodes, probe, decoder
  probe.py              linear probe: latent -> the task's angles / positions
  decoder.py            pixel decoder on the frozen latents (for looking only)
  rollout.py            copy / teacher-forced / autoregressive / shuffled-action rollouts
src/
  tasks.py              per task: what the state means and what the probe reads
  data/                 HDF5 reader, training windows, z-score stats, pixel transform
  models/               encoders (ViT, DINOv2, state MLP), predictor, SIGReg, LeWM
  utils.py              seeding, schedules, logging, checkpoints, BatchNorm recalibration
checkpoints/<task>/<preset>/   epoch_NNN.pt, train.log, and the analyses: probe/ decoder/ rollout/
temp/                   files moved out of the way (not tracked)
```

## Setup

```bash
conda activate temporal
pip install -r requirements.txt
```

## Run a task

```bash
hf download nicklashansen/mmbench2 --repo-type dataset --local-dir /home/world-models/data \
  --include "*/acrobot-swingup.pt"  "*/acrobot-swingup-[0-9]*.png" \
            "*/cartpole-swingup.pt" "*/cartpole-swingup-[0-9]*.png" \
            "*/pendulum-swingup.pt" "*/pendulum-swingup-[0-9]*.png" \
            "*/reacher-easy.pt"     "*/reacher-easy-[0-9]*.png"

python run.py --task acrobot-swingup                    # everything, ViT, up to epoch_006.pt
python run.py --task cartpole-swingup --stop-epoch 9
python run.py --task reacher-easy --stages probe rollout --force   # redo some analyses
python run.py --task pendulum-swingup --smoke           # few-minute end-to-end check
python run.py --task cube-single                        # OGBench cube (LeWM's data, no conversion)

python ablations/compare_rollouts.py checkpoints/cartpole-swingup/vit checkpoints/cube-single/vit \
    --out ../rollout_comparison.png                     # one figure, a row per task
```

`run.py` runs, for one task, and skips any stage whose output exists:

1. **data**: `datasets/prepare_mmbench.py` writes the training table (expert +
   mixed-small + mixed-large + zeros, 260 episodes of 501 rows) and the val + test
   table (40 episodes the model never sees). It keeps only the real action and
   state columns (MMBench pads every task to 16 / 128) and shifts actions so row t
   holds the action taken *at* t.
2. **train**: `train.py`, epochs 0..`--stop-epoch` (default 6) of 2000 steps into
   `checkpoints/<task>/<preset>/`, on the LR schedule of a 100-epoch run (as the
   first acrobot run, stopped by hand at epoch 6); a partial run resumes.
3. **probe**: `ablations/probe.py`, a closed-form ridge from the latent of
   `epoch_<stop-epoch>.pt` to the task's state (src/tasks.py), scored on the 26 held-out episodes,
   plus the same probe on the untrained architecture (`probe_random/`).
4. **decoder**: `ablations/decoder.py`, a conv decoder from the latent to a 112x112
   frame, trained on frozen latents; only used to look at rollouts.
5. **rollout**: `ablations/rollout.py`, per horizon step, latent MSE and probe error
   for copy-last, teacher-forced, autoregressive (recorded actions) and
   autoregressive with another episode's actions, on held-out + val/test episodes;
   plus the probe on real frames as a floor and the accumulated error
   (autoregressive minus teacher-forced). Videos: truth | decode(enc(truth)) |
   teacher-forced | autoregressive | shuffled.

Every stage is also a plain script, e.g.
`python train.py --preset vit data.task=cartpole-swingup` or
`python ablations/probe.py --ckpt checkpoints/cartpole-swingup/vit/epoch_006.pt`.
Presets: `vit` (LeWM, ViT-tiny from scratch + SIGReg), `dino` (frozen DINOv2
patches, no SIGReg), `oracle` (MLP on the true state), `debug`.

## Experiments

Four DMControl tasks from MMBench2 (`nicklashansen/mmbench2`), each recorded by
TD-MPC2 agents. Per task, the training table holds the `expert`, `mixed-small`,
`mixed-large` and `zeros` splits: **260 episodes of 501 rows** (130,260 frames),
26 of them held out by the world model; `val` + `test` (40 more episodes) are
never trained on. One row is **2 simulator steps** (TD-MPC2's action repeat, so
per-row reward reaches 2). Frames are 224x224 RGB.

| task | observation (oracle state) | action | probe reports |
|---|---|---|---|
| acrobot-swingup | 6: cos upper, cos lower, sin upper, sin lower link; 2 joint velocities | 1: torque at the **elbow** (the shoulder has no motor) | shoulder, elbow angle (deg) |
| cartpole-swingup | 5: cart x; cos, sin pole; cart velocity; pole angular velocity | 1: horizontal **force on the cart** | cart position (cm), pole angle (deg) |
| pendulum-swingup | 3: cos, sin pole; angular velocity | 1: **torque at the pivot**, too weak to lift the pole directly (it has to swing) | pole angle (deg) |
| reacher-easy | 6: shoulder angle, wrist angle (raw radians); finger-to-target x, y; 2 joint velocities | 2: torques at the **shoulder** and the **wrist** | shoulder, wrist (deg), to_target x, y (cm) |
| cube-single | 28: arm joints (pos, vel), effector xyz + yaw, gripper opening + contact, cube xyz + quaternion + yaw | 5: effector **x, y, z, yaw** and **gripper** command | cube, effector position (3-D distance, cm) |

**cube-single** is the OGBench recording LeWM itself trains on
(`.stable_worldmodel/datasets/ogbench/cube_single_expert.h5`, used as is):
10,000 expert episodes of 201 rows (41 latent frames), 1,000 held out, no
separate val/test file. Its rollouts use a 30-step horizon and its probe and
decoder every 10th training row (`TASK_ARGS` in run.py, `TASK_DATA` in config.py).

- Every action is normalised to [-1, 1]. MMBench pads actions to 16 columns
  (noise in the `mixed`/`val`/`test` splits); only the real ones are kept.
- The probe regresses angles as cos/sin (a raw angle jumps at +-180 degrees, and
  reacher's shoulder winds past it), positions as is (src/tasks.py).
- The world model never sees the oracle state with the `vit` preset; it is only
  the probe's target.

## Hyperparameters

All in `config.py` (preset `vit`) and the scripts' defaults; `run.py` sets only
`data.task` and `optim.stop_epoch`.

**Data**

| | |
|---|---|
| frameskip | 5 rows (10 sim steps) per latent step; the action of a latent step is its 5 raw actions, flattened (5 x action_dim) |
| history | 3 latent frames fed to the predictor (also its context at inference) |
| training window | history 3 + 1 predicted = 4 latent frames = 20 rows |
| image size | 224, ImageNet normalisation |
| window stride | train 2 rows, val 5 rows |
| held-out | 26 of 260 episodes (seeded split, seed 3072) |
| normalisation | z-score of actions (and state for the oracle), cached next to the h5 |

**Model** (18.1M parameters, all trained jointly)

| | |
|---|---|
| encoder | ViT-tiny/16 (`vit_tiny_patch16_224`) from scratch, class token, width 192 |
| projector, pred_proj | MLP 192 -> 2048 -> 192 with BatchNorm |
| action encoder | 1x1 conv to 64, then MLP 64 -> 768 -> 192 |
| predictor | causal transformer, 6 layers, 16 heads x 64, MLP 2048, dropout 0.1, actions via AdaLN-zero |
| loss | next-latent MSE (teacher-forced) + 0.09 x SIGReg (17 knots, 1024 projections) |

**Optimisation**

| | |
|---|---|
| optimiser | AdamW, lr 5e-5, weight decay 1e-3, betas (0.9, 0.999), grad clip 1.0, bf16 autocast |
| batch | 128 windows |
| epoch | 2000 steps (random windows with replacement), validation 100 steps |
| LR schedule | 2 warmup epochs, then cosine to 0.01 x lr over `optim.epochs` = 100 epochs |
| stopping | after epoch `optim.stop_epoch` = 6 (7 x 2000 steps); everything is analysed on `epoch_006.pt` |
| time | ~14.5 min per epoch on one 24 GB GPU, ~1h45m per task |

**Analyses**

| | |
|---|---|
| probe | closed-form ridge (strength 1e-3 x N) on standardised features; every 2nd training row; patch latents pooled to 4x4 |
| decoder | conv decoder to 112x112, 20k steps, batch 128, AdamW lr 1e-3 (one-cycle), 10x loss weight on pixels away from the mean frame; every 2nd training row |
| rollout | horizon 50 latent steps (250 rows), a window at every latent step, held-out + val/test episodes (66), 4 videos |

## Notes

- **BatchNorm in eval mode.** The projector heads use BatchNorm, and their running
  statistics (collected with dropout on) do not fit eval mode: held-out
  prediction loss was ~9x too high. `src.utils.recalibrate_bn` recomputes them;
  train.py does it before every validation pass, and the analyses do it on load.
  DINO runs have no projector, so nothing to do there.
- **Frozen DINO.** The encoder never changes, so the DINO probe and decoder do not
  depend on how long the predictor trained; only the rollouts do.
- **Shapes.** Latents are (B, T, P, D): P = 1 for the ViT class token, 196 for DINO
  patches. Latent step t is row t * frameskip; action block t (frameskip raw
  actions) carries step t to t + 1.
