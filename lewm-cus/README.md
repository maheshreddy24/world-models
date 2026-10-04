# lewm-cus

A clean, modular reimplementation of **LeWM** (LeWorldModel): a joint-embedding
predictive world model trained end-to-end from pixels, and planned with on two
OGBench tasks — **cube-single** (MPC) and **scene-play** (a diffusion planner on
seven manipulation tasks). `data.backend` picks the recording; everything above
the data layer is shared.

The whole thing is two loss terms and one hyperparameter — predict the next
embedding, and keep the embedding distribution Gaussian so it cannot collapse.
No EMA target, no stop-gradient, no pretrained encoder, no decoder.

## Layout

```
config.py          every knob, as dataclasses + presets. no hydra.
train.py           training loop (stage 1: the dynamics predictor)
train_policy.py    diffusion policy on a frozen world model (stage 2)
eval.py            closed-loop planning on cube-single -> success rate + videos
eval_scene.py      closed-loop evaluation on the seven scene v1 tasks
rollout.py         open-loop latent rollout -> error curves + "imagination" videos
datasets/          data generation / preparation (run once)
  prepare_scene.py   inflate the scene .npz into a memory-mappable store
  make_v1_pairs.py   mine the held-out scene evaluation pairs
  scene_v1.py        mine the scene training pairs
  make_ball_data.py  generate the BallCatch HDF5 recording
ablations/         probes (probe_cube/ball/ball_px), latent analysis, oracle sweeps
temp/              old outputs no longer in use
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
    lewm_diffusion_policy.py  goal-conditioned action diffusion + its planner
  data/
    backends.py    the one place that knows which recording is in play
    ogbench.py     cube-single: HDF5 sequence dataset + evaluation task sampling
    scene_store.py scene-play: npz -> memory-mapped store, and the reader
    scene.py       scene-play: goal-conditioned pair dataset for the planner
    scene_tasks.py the seven v1 tasks: features, mining, success check
    normalize.py   cached z-score statistics
    transforms.py  pixel preprocessing
  envs/
    cube.py        OGBench cube environments, stepped in lockstep
    scene.py       OGBench scene environments, stepped in lockstep
  utils.py         seeding, schedules, checkpoints, logging, video
```

Both recordings go through the same three functions — `build_datasets`,
`build_policy_datasets`, `open_store` — so `train.py`, `train_policy.py` and
`rollout.py` never name a reader.

## Setup

Needs `torch`, `timm`, `h5py`, `hdf5plugin`, `einops`, `gymnasium`, `mujoco`,
`imageio`, `matplotlib`, `tqdm`, and optionally `wandb`. Evaluation also needs
the env package for whichever task you run: `stable-worldmodel` registers
`swm/OGBCube-v0` for the cube, `ogbench` registers `visual-scene-v0` for the
scene. Neither is imported until the matching wrapper is used.

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

## Diffusion policy (alternative planner)

A goal-conditioned Diffusion Policy (the WorldDP low level) on top of a trained,
frozen world model. It denoises the next `policy.chunk` (5) raw actions under

```
c = [ z_t      frozen world-model latent of the current frame (pixel latent or oracle state)
      proprio  joint pos 6 + joint vel 6 + gripper opening 1        policy.proprio=arm (13)
      goal     the cube's target xyz, metres                         policy.goal=cube_xyz (3)
      contact  gripper contact, optional                             policy.use_contact=true (1) ]
```

each part its own token in the denoiser. The world model itself is unchanged;
its side of the contact add-on is `model.drop_obs=gripper_contact`. The earlier
conditioning (goal-frame latent + end-effector position/velocity) is
`policy.goal=latent policy.proprio=ee`, and older policy checkpoints load as
that. The same policy is the planner on scene-play, with latent goals drawn from
the mined v1 pairs instead — see [OGBench scene-play](#ogbench-scene-play-databackendscene).

```bash
python train_policy.py --ckpt <world model run>/epoch_007.pt
python train_policy.py --ckpt <...> policy.use_contact=true
python eval.py plan.planner=diffusion --diffusion-ckpt <policy run>/epoch_019.pt
python eval.py plan.planner=diffusion --diffusion-ckpt <...> plan.diffusion_samples=16
```

Training goals are drawn 5-50 raw steps ahead, so eval's 25-step goals are in
distribution. Data, model and the episode split come from the world-model
checkpoint, and the policy checkpoint records which world model it belongs to,
so eval needs only `--diffusion-ckpt`. With `plan.diffusion_samples > 1` the
world model rolls every sampled chunk out and executes the one that lands
closest to the goal frame's latent, whichever goal the policy itself reads.
Watch `val/action_mse` while training.

## OGBench scene-play (`data.backend=scene`)

The second dataset: 2,000 episodes x 1,001 steps of 64x64 frames on the OGBench
Scene, with a drawer, a window, two buttons and a cube. Seven **v1 tasks** are
mined out of it — `open_drawer`, `close_drawer`, `open_window`, `close_window`,
`move_cube`, `cube_into_drawer`, `toggle_lock` — as (start frame, goal frame)
pairs in which exactly one thing about the scene has changed.

```
ogbench_scene_single/
  visual-scene-play-v0.npz       training recording, 2,002,000 rows
  visual-scene-play-v0-val.npz   validation recording, 200,200 rows
  train_pairs.npz                187k mined pairs (indices into the training recording)
  v1_pairs.npz                   350 evaluation pairs, 50 per task, self-contained
```

### 0. Prepare the store (once)

```bash
python datasets/prepare_scene.py --effector
```

The frames are deflated inside the `.npz` and deflate cannot seek, so a
dataloader would have to inflate 1.5M rows to reach row 1,500,000. This inflates
each column once into a `.npy` the dataloader memory-maps: about 25 GB on disk
and a few minutes, at a few hundred MB of peak RAM. `--effector` adds
`effector_pos`, the gripper's world position, which the recording does not store
— it is recovered from `qpos` by forward kinematics (seconds, and exact) and is
what the policy's proprio is built from.

### 1. The dynamics predictor

```bash
python train.py --preset scene_pixels                 # ViT-tiny at native 64x64
python train.py --preset scene_dino                   # frozen DINOv2 patches, 64 -> 224
python train.py --preset scene_debug                  # smoke test
```

Frames are 64x64, so the ViT runs at 64 and sees a 4x4 patch grid; nothing is
upsampled. Validation is the recording's own held-out file, not a slice of the
training episodes.

### 1b. The oracle world model

The same two stages with the ViT swapped for an MLP over the scene *state*.
Where the pixel run and the oracle run disagree tells you whether a failing task
is the representation's fault or the planner's.

```bash
python datasets/prepare_scene.py --state                       # once: the 39-d state column
python train.py --preset scene_oracle
python train_policy.py --preset scene_policy --ckpt checkpoints/<oracle run>/epoch_0NN.pt
python eval_scene.py plan.planner=diffusion --diffusion-ckpt checkpoints/<oracle policy>/epoch_0NN.pt
```

The recordings store `qpos`/`qvel`/`button_states` but not OGBench's state
observation, so `--state` rebuilds it (`src/data/scene_state.py`). 39 of its 40
entries are exact functions of those arrays; `gripper_contact` is a contact
force MuJoCo only computes inside a physics step and is dropped. Training and
evaluation both call the same `SceneState` function.

`v1_pairs.npz` stores goal positions but not goal velocities, so evaluation
goals are built with zero velocity — and stage 2 zeroes the velocity entries of
its training goals to match. Stage 1 keeps real velocities: dynamics need them.

### 2. The diffusion planner

```bash
python train_policy.py --preset scene_policy --ckpt checkpoints/<scene run>/epoch_019.pt
python eval_scene.py plan.planner=diffusion --diffusion-ckpt checkpoints/<policy run>/epoch_019.pt
python eval_scene.py --policy random                  # the floor
python eval_scene.py --ckpt checkpoints/<scene run>/epoch_019.pt   # MPC, for comparison
```

The policy denoises the next `policy.chunk` (15) raw actions from the current
latent, the goal latent and the proprio, and the planner replans every chunk —
closed loop, so it recovers from drift over the 30-160 step gaps the tasks span.

Where its goals come from is the whole design, and it lives in
`src/data/scene.py`:

| knob | what it does |
| --- | --- |
| `scene.p_task` | fraction of draws that use a mined v1 pair; the rest are hindsight goals (any frame, a later frame of the same episode) |
| `scene.balance_tasks` | sample the seven tasks equally. `toggle_lock` has 80,385 mined pairs and `cube_into_drawer` 3,244, so uniform sampling over pairs would almost never show the rare one |
| `scene.resample_start` | redraw the start row inside `[start, goal - chunk]`. Training only at the mined start teaches the first move of each task and nothing about recovering half-way through; this is the state distribution closed-loop replanning actually visits |
| `scene.hindsight_min_gap` / `max_gap` | how far ahead a hindsight goal sits. `min_gap` must be at least `policy.chunk` |

### Evaluation

`v1_pairs.npz` is self-contained: each row carries the simulator state to reset
to, the state success is measured against, and both frames. Evaluation never
opens a recording, and the `start_idx`/`goal_idx` columns are provenance from
whichever recording the pairs were mined from — **not** an index into the one you
are training on. Mine the evaluation pairs from a recording you do not train on.

Tasks run in batches of `scene.eval_batch` (50) on one pool of envs that is
built once and reused. Each env holds its own EGL render context at ~124 MB of
GPU memory, so all 350 at once would need ~43 GB — more than an L4 has.

Success is task-agnostic (`src/data/scene_tasks.py:is_success`): the final cube
pose, drawer, window and buttons must match the goal state, and the arm is
ignored. `eval_scene.py` reports it per task as well as overall, and prints
`start_frame_mean_abs_diff` — how far the rendered start frame is from the
recorded one. It should be a fraction of a grey level; anything large means the
encoder is being shown frames the recording never contained.

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
| a new recording | add a store + a `build_*` pair under `src/data/`, then one line per table in `src/data/backends.py` |

`model.proj_norm` defaults to `"batch"` to match the reference implementation.
BatchNorm makes the embedding batch-dependent while training and
running-statistic-dependent at eval, so `val/emb_std` lags `train/emb_std` until
the running stats settle. Set it to `"layer"` for identical train/eval behaviour.
