# lewm-cus

A clean, modular reimplementation of **LeWM** (LeWorldModel): a joint-embedding
predictive world model trained end-to-end from pixels, and planned with on the
OGBench cube task.

The whole thing is two loss terms and one hyperparameter — predict the next
embedding, and keep the embedding distribution Gaussian so it cannot collapse.
No EMA target, no stop-gradient, no pretrained encoder, no decoder.

## Layout

```
config.py          every knob, as dataclasses + presets. no hydra.
train.py           training loop
eval.py            closed-loop planning in the simulator -> success rate + videos
rollout.py         open-loop latent rollout -> error curves + "imagination" videos
src/
  models/
    encoder.py     ViT-tiny (timm) for pixels, MLP for oracle states
    predictor.py   action encoder + autoregressive latent dynamics
    blocks.py      attention / AdaLN-zero transformer blocks
    sigreg.py      the anti-collapse regulariser
    lewm.py        the world model: encode, predict, rollout, planning cost
  planner/
    solvers.py     CEM and MPPI over action sequences
    mpc.py         receding-horizon control loop
  data/
    ogbench.py     HDF5 sequence dataset + evaluation task sampling
    normalize.py   cached z-score statistics
    transforms.py  pixel preprocessing
  envs/cube.py     OGBench cube environments, stepped in lockstep
  utils.py         seeding, schedules, checkpoints, logging, video
```

## Setup

Needs `torch`, `timm`, `h5py`, `hdf5plugin`, `einops`, `gymnasium`,
`stable-worldmodel` (used only to register the `swm/OGBCube-v0` env),
`imageio`, `matplotlib`, `tqdm`, and optionally `wandb`.

Point `config.py` at the data: it looks for `.stable_worldmodel/datasets/` next
to the repo, then `$STABLEWM_HOME`, then `~/.stable_worldmodel`. Override with

```bash
export STABLEWM_HOME=/path/to/store        # or: python train.py data.h5_path=/path/to/file.h5
```

## Configuration

One file, `config.py`. Change a default there, pick a preset, or override on the
command line — overrides are dotted paths cast to the field's type:

```bash
python train.py                              # pixels, defaults
python train.py --preset cube_oracle         # MLP over the 28-d state, trains in minutes
python train.py --preset debug               # 20-second smoke test
python train.py optim.lr=1e-4 model.depth=8 wandb.enabled=true
```

Presets live in `PRESETS` at the bottom of `config.py`; add your own as a dict of
dotted overrides.

## Train

```bash
python train.py --preset cube_oracle         # sanity-check the whole loop first
python train.py                              # the real thing, from pixels
python train.py --resume                     # continue from <run_dir>/last.pt
```

Checkpoints, `config.json` and `metrics.csv` land in
`<out_dir>/<run_name>/`. The config travels inside the checkpoint, so `eval.py`
and `rollout.py` only need `--ckpt`.

Two numbers tell you if it is working:

| metric | meaning |
| --- | --- |
| `emb_std` | collapse detector. Heading for 0 means the representation is dying and `loss.sigreg_weight` is too low. |
| `pred_vs_static` | prediction error over the "nothing moves" baseline. Below 1 means the predictor has learned dynamics. |

An epoch is `optim.steps_per_epoch` randomly-drawn windows rather than a full
pass — the dataset holds millions of overlapping windows, so a literal epoch
would produce one checkpoint a few hours apart.

## Evaluate (closed-loop planning)

```bash
python eval.py --ckpt <run_dir>/best.pt
python eval.py --ckpt <run_dir>/best.pt plan.solver=mppi plan.num_samples=500
python eval.py --policy random                 # the baseline to beat
```

Each task replays a recorded start state and asks the planner to reach the pose
the expert reached `eval.goal_offset` steps later, given only the goal **image**.
Success is the simulator's own check (cube within 4cm of target). Writes
`results.json` and side-by-side videos (agent | expert | goal).

Evaluation runs on held-out episodes by default (`--split val`).

**One thing worth knowing about the task.** The expert spends much of each
episode reaching toward the cube before it ever moves it, so over a 25-step
window the cube sits inside its own goal tolerance about 40% of the time. Those
tasks are already solved before the planner acts, and a random policy "succeeds"
at them — scoring 31% instead of its true ~5%. `eval.min_goal_distance`
(default 0.10m) drops them, so the success rate means what it says. Set it to 0
to reproduce the unfiltered number.

## Rollout (open-loop diagnostics)

```bash
python rollout.py --ckpt <run_dir>/best.pt rollout.horizon=20
```

Gives the model the first few observations plus the expert's actions and lets it
predict the rest of the latent trajectory on its own output. Produces:

* per-horizon latent error against the encoder's own view of the future, next to
  a static baseline — where the ratio crosses 1 is roughly how far ahead it is
  worth planning (`plan.horizon`);
* "imagination" videos: each predicted latent matched to its nearest real frame,
  which is how you inspect a rollout when there is no decoder.

## Conventions

`B` batch, `T` latent steps, `D` embedding, `S` planning samples.

One **latent step** spans `data.frameskip` raw env steps. The action of a latent
step is the flattened block of the raw actions it covers, so the model sees
`action_dim * frameskip` numbers per step and one planning step commits to a
block of env actions. `plan.action_block` must equal `data.frameskip`;
`config.validate()` enforces it.

The solver optimises in **normalised** action space (the same z-scoring used in
training) and actions are denormalised and clipped to the env's box on the way
out.

## Swapping pieces

| want to change | where |
| --- | --- |
| encoder (bigger ViT, ResNet, ...) | `src/models/encoder.py`, then `model.vit_name` + `model.embed_dim` |
| dynamics model | `src/models/predictor.py` |
| planning cost | `LeWM.goal_cost` in `src/models/lewm.py`, selected by `model.cost` |
| a new solver | subclass `Solver` in `src/planner/solvers.py`, register it in `build_solver` |
| a different env | add a wrapper in `src/envs/` with the same `reset_to` / `step` contract |

`model.proj_norm` defaults to `"batch"` to match the reference implementation.
BatchNorm makes the embedding batch-dependent while training and
running-statistic-dependent at eval, so `val/emb_std` lags `train/emb_std` until
the running stats settle. Set it to `"layer"` for identical train/eval behaviour.
