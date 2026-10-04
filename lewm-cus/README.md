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

What the probe reads, per task:

| task | probe targets | reported |
|---|---|---|
| acrobot-swingup | cos/sin of both links | shoulder, elbow angle (deg) |
| cartpole-swingup | cart x, cos/sin pole | cart position (cm), pole angle (deg) |
| pendulum-swingup | cos/sin pole | pole angle (deg) |
| reacher-easy | cos/sin of both joints, finger-to-target x/y | shoulder, wrist (deg), to_target x/y (cm) |

Every stage is also a plain script, e.g.
`python train.py --preset vit data.task=cartpole-swingup` or
`python ablations/probe.py --ckpt checkpoints/cartpole-swingup/vit/epoch_006.pt`.
Presets: `vit` (LeWM, ViT-tiny from scratch + SIGReg), `dino` (frozen DINOv2
patches, no SIGReg), `oracle` (MLP on the true state), `debug`.

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
