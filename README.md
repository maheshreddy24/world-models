# world-models

> 🚧 **Work in progress.** Code, configs and results change often; numbers below are early and unreviewed.

Experiments with latent world models for planning and control, built around
**LeWM** (LeWorldModel) — a JEPA trained end-to-end from pixels with a
next-embedding prediction loss plus the SIGReg anti-collapse regulariser.

The main code lives in [`lewm-cus/`](lewm-cus/), a modular reimplementation of
LeWM with MPC (CEM / MPPI) and diffusion-policy planners. See
[`lewm-cus/README.md`](lewm-cus/README.md) for layout, training and evaluation.

## Current experiments

- **OGBench cube-single / cube-double** — closed-loop goal-image planning with MPC.
- **OGBench scene-play** — diffusion-policy planner over seven mined tasks, with a state-based oracle world model for comparison.
- **Acrobot swing-up (MMBench)** — double pendulum from pixels. Early probes read both joint angles linearly off the frozen latent (~1° / ~2° MAE held-out); open-loop rollouts and decoders under way.
- **BallCatch** — small 2D pymunk catching task for probing what the latent encodes.
- **Ablations** — pixel vs. oracle-state vs. frozen DINOv2 encoders, linear probes, latent readability.

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
