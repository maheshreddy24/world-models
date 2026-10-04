# world-models

> 🚧 **Work in progress.** Code, configs and results change often; numbers below are early and unreviewed.

Experiments with latent world models for planning and control, built around
**LeWM** (LeWorldModel) — a JEPA trained end-to-end from pixels with a
next-embedding prediction loss plus the SIGReg anti-collapse regulariser.

The main code lives in [`lewm-cus/`](lewm-cus/), a small reimplementation of
LeWM. See
[`lewm-cus/README.md`](lewm-cus/README.md) for layout, training and evaluation.

## Current experiments

LeWM (ViT-tiny trained from scratch) on four DMControl tasks from MMBench2,
one task at a time. For each: train the world model, fit a linear probe from
the frozen latent to the true state, train a pixel decoder for visualisation,
then measure how prediction error compounds over autoregressive rollouts
against copy-last, teacher-forced and shuffled-action baselines.

```bash
cd lewm-cus && python run.py --task cartpole-swingup    # data -> train -> probe -> decoder -> rollout
```

| task | observation (true state) | action | probe reports |
|---|---|---|---|
| acrobot-swingup | 6: cos/sin of both links, 2 joint velocities | 1: torque at the **elbow** (the shoulder has no motor) | shoulder, elbow angle (deg) |
| cartpole-swingup | 5: cart x, cos/sin pole, cart velocity, pole angular velocity | 1: horizontal **force on the cart** | cart position (cm), pole angle (deg) |
| pendulum-swingup | 3: cos/sin pole, angular velocity | 1: **torque at the pivot**, too weak to lift the pole directly | pole angle (deg) |
| reacher-easy | 6: shoulder, wrist angle; finger-to-target x, y; 2 joint velocities | 2: torques at the **shoulder** and **wrist** | shoulder, wrist (deg), to_target x, y (cm) |

- **Data per task:** 260 training episodes of 501 rows (expert, mixed-small,
  mixed-large, zeros), 26 held out; 40 more (val + test) never trained on.
  224x224 frames; one row = 2 simulator steps; actions in [-1, 1].
- **Model:** ViT-tiny/16 encoder (192-d class token) + BatchNorm MLP projectors +
  6-layer causal transformer predictor (16 heads, MLP 2048), 18.1M parameters,
  all trained jointly. Loss: next-latent MSE + 0.09 x SIGReg.
- **Training:** frameskip 5, history 3 latent frames, batch 128, AdamW lr 5e-5
  (weight decay 1e-3, 2 warmup epochs, cosine schedule of a 100-epoch run),
  2000 steps per epoch, stopped after epoch 6; every analysis uses `epoch_006.pt`.
- **Rollout ablation:** horizon 50 latent steps (250 rows) from every step of the
  66 unseen episodes; latent MSE and probe error per horizon step.

Full hyperparameter tables are in [`lewm-cus/README.md`](lewm-cus/README.md#hyperparameters).
A first acrobot run (`checkpoints/double_pend_lwm`) read both joint angles off the
frozen latent at ~1° / ~2° mean error on held-out episodes. DINO-WM (frozen DINOv2
patches) and the true state are supported as encoder comparisons (`--preset dino`, `oracle`).

The earlier OGBench cube / scene and BallCatch pipelines were removed from the codebase; they are in git history (commit `7f5107c`).

## Setup

```bash
bash setup.sh                         # conda env + PyTorch
pip install -r lewm-cus/requirements.txt
```

## References

**LeWM**, the model this repo reimplements:

```bibtex
@article{maes_lelidec2026lewm,
  title   = {LeWorldModel: Stable End-to-End Joint-Embedding Predictive Architecture from Pixels},
  author  = {Maes, Lucas and Le Lidec, Quentin and Scieur, Damien and LeCun, Yann and Balestriero, Randall},
  journal = {arXiv preprint arXiv:2603.19312},
  year    = {2026}
}
```

Methods
- Balestriero & LeCun. *LeJEPA: Provable and Scalable Self-Supervised Learning Without the Heuristics.* 2025. [arXiv:2511.08544](https://arxiv.org/abs/2511.08544) — SIGReg
- Assran et al. *Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture (I-JEPA).* CVPR 2023. [arXiv:2301.08243](https://arxiv.org/abs/2301.08243)
- Zhou, Pan, LeCun & Pinto. *DINO-WM: World Models on Pre-trained Visual Features enable Zero-shot Planning.* 2024. [arXiv:2411.04983](https://arxiv.org/abs/2411.04983)
- Oquab et al. *DINOv2: Learning Robust Visual Features without Supervision.* 2023. [arXiv:2304.07193](https://arxiv.org/abs/2304.07193)
- Dosovitskiy et al. *An Image is Worth 16x16 Words (ViT).* ICLR 2021. [arXiv:2010.11929](https://arxiv.org/abs/2010.11929)
- Chi et al. *Diffusion Policy: Visuomotor Policy Learning via Action Diffusion.* RSS 2023. [arXiv:2303.04137](https://arxiv.org/abs/2303.04137)
- *Unifying Object-Centric World Models and Diffusion Policy: A Hierarchical Framework for Multi-Stage Robotic Tasks (WorldDP).* 2026. [arXiv:2606.08775](https://arxiv.org/abs/2606.08775)
- Rubinstein. *The Cross-Entropy Method for Combinatorial and Continuous Optimization.* 1999 — CEM
- Williams et al. *Information Theoretic MPC for Model-Based Reinforcement Learning.* ICRA 2017 — MPPI

Benchmarks & data
- Park et al. *OGBench: Benchmarking Offline Goal-Conditioned RL.* 2024. [arXiv:2410.20092](https://arxiv.org/abs/2410.20092)
- Hansen, Su & Wang. *Learning Massively Multitask World Models for Continuous Control (Newt / MMBench).* ICLR 2026. [arXiv:2511.19584](https://arxiv.org/abs/2511.19584)
- Hansen, Su & Wang. *TD-MPC2: Scalable, Robust World Models for Continuous Control.* ICLR 2024. [arXiv:2310.16828](https://arxiv.org/abs/2310.16828)

Software
- [stable-worldmodel](https://github.com/galilai-group/stable-worldmodel),
  [timm](https://github.com/huggingface/pytorch-image-models),
  [MuJoCo](https://mujoco.org/) (Todorov et al., IROS 2012),
  [pymunk](https://www.pymunk.org/),
  [Weights & Biases](https://wandb.ai/)
