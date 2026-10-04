"""Goal-conditioned action diffusion on top of a frozen LeWM (the WorldDP low-level policy).

The policy denoises the next `chunk` raw env actions conditioned on

    z_cur    LeWM latent of the current frame                           (P, D)
    goal     policy.goal: cube target xyz in metres (3,), or the LeWM
             latent of the goal frame (P, D)
    proprio  policy.proprio: joint pos + joint vel + gripper opening
             (13,), or end-effector position + per-step velocity (6,)   optional
    contact  gripper contact                                            (1,)   optional

The denoiser is the transformer variant of Diffusion Policy (Chi et al.): the
diffusion timestep and the conditions become memory tokens for a small
encoder, and the noisy action tokens go through a causal decoder that
cross-attends to them.  Latents enter as tokens, so a pooled world model
(P = 1) and a DINO patch world model (P = 196) both work unchanged; every
vector condition is one token.

`DiffusionPlanner` wraps a trained policy for eval.py with the same interface
as `MPCPlanner`.  With `plan.diffusion_samples > 1` it draws several chunks and
lets the world model keep the one whose predicted outcome is closest to the goal.

Conventions: B batch, H chunk (raw env steps), A raw action dim,
P tokens per frame, D latent width, S samples per env.
"""

from __future__ import annotations

import math
from collections import deque

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .mpc import to_model_input

# Widths of the vector conditions per config value; see src.data.ogbench.cube_proprio.
PROPRIO_DIMS = {"arm": 13, "ee": 6}
GOAL_DIMS = {"cube_xyz": 3, "latent": 0}  # 0: the goal is a latent frame, P tokens of width D


# --------------------------------------------------------------------------- #
#  Diffusion utilities
# --------------------------------------------------------------------------- #
def cosine_alphas_cumprod(steps: int, s: float = 0.008) -> torch.Tensor:
    """Cumulative signal fraction of the cosine noise schedule (Nichol & Dhariwal): (steps,)."""
    t = torch.linspace(0, steps, steps + 1) / steps
    f = torch.cos((t + s) / (1 + s) * math.pi / 2) ** 2
    alphas_cumprod = f / f[0]
    betas = (1 - alphas_cumprod[1:] / alphas_cumprod[:-1]).clamp(max=0.999)
    return torch.cumprod(1 - betas, dim=0)


def timestep_embedding(k: torch.Tensor, dim: int) -> torch.Tensor:
    """Sinusoidal embedding of integer timesteps: (B,) -> (B, dim)."""
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=k.device) / half)
    angles = k.float()[:, None] * freqs[None]
    return torch.cat([angles.sin(), angles.cos()], dim=-1)


# --------------------------------------------------------------------------- #
#  Denoiser
# --------------------------------------------------------------------------- #
class Denoiser(nn.Module):
    """Predict the noise in an action chunk.

    Memory tokens: [timestep, z_cur x P, goal, proprio, contact], each with a
    learned position.  The goal is P tokens when it is a latent frame
    (`goal_dim=0`) and one token when it is a `goal_dim` vector; proprio and
    contact are one token each when present.  The decoder is causal over the
    H action tokens.
    """

    def __init__(
        self,
        action_dim: int,
        latent_dim: int,
        num_tokens: int,
        chunk: int,
        goal_dim: int = 0,
        proprio_dim: int = 0,
        contact_dim: int = 0,
        width: int = 256,
        depth: int = 8,
        heads: int = 4,
        cond_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.width = width
        self.action_in = nn.Linear(action_dim, width)
        self.action_pos = nn.Parameter(torch.randn(1, chunk, width) * 0.02)

        self.time_mlp = nn.Sequential(nn.Linear(width, 4 * width), nn.Mish(), nn.Linear(4 * width, width))
        self.cur_in = nn.Linear(latent_dim, width)
        self.goal_in = nn.Linear(goal_dim or latent_dim, width)
        self.proprio_in = nn.Linear(proprio_dim, width) if proprio_dim else None
        self.contact_in = nn.Linear(contact_dim, width) if contact_dim else None
        num_cond = 1 + num_tokens + (1 if goal_dim else num_tokens) + bool(proprio_dim) + bool(contact_dim)
        self.cond_pos = nn.Parameter(torch.randn(1, num_cond, width) * 0.02)

        enc_layer = nn.TransformerEncoderLayer(width, heads, 4 * width, dropout, batch_first=True, norm_first=True)
        dec_layer = nn.TransformerDecoderLayer(width, heads, 4 * width, dropout, batch_first=True, norm_first=True)
        self.cond_encoder = nn.TransformerEncoder(enc_layer, cond_layers, enable_nested_tensor=False)
        self.decoder = nn.TransformerDecoder(dec_layer, depth)
        self.out = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, action_dim))
        causal = torch.triu(torch.full((chunk, chunk), float("-inf")), diagonal=1)
        self.register_buffer("causal_mask", causal, persistent=False)

    def forward(
        self,
        noisy_action: torch.Tensor,
        k: torch.Tensor,
        z_cur: torch.Tensor,
        goal: torch.Tensor,
        proprio: torch.Tensor | None = None,
        contact: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """noisy_action: (B, H, A), k: (B,), z_cur: (B, P, D), goal: (B, P, D) or (B, goal_dim),
        proprio: (B, proprio_dim), contact: (B, contact_dim) -> noise (B, H, A)"""
        cond = [
            self.time_mlp(timestep_embedding(k, self.width))[:, None],
            self.cur_in(z_cur),
            self.goal_in(goal if goal.dim() == 3 else goal[:, None]),
        ]
        if self.proprio_in is not None:
            cond.append(self.proprio_in(proprio)[:, None])
        if self.contact_in is not None:
            cond.append(self.contact_in(contact)[:, None])
        memory = self.cond_encoder(torch.cat(cond, dim=1) + self.cond_pos)
        x = self.action_in(noisy_action) + self.action_pos
        return self.out(self.decoder(x, memory, tgt_mask=self.causal_mask))


# --------------------------------------------------------------------------- #
#  Policy: DDPM training loss and DDIM sampling
# --------------------------------------------------------------------------- #
class DiffusionPolicy(nn.Module):
    """Denoiser plus the normalisation it needs, so a checkpoint is self-contained.

    Actions are min-max scaled to [-1, 1] (the DDIM sampler clips its x0 guess
    there); proprio, a vector goal and contact are z-scored.  All statistics are
    buffers filled by `set_stats` from the training data and saved with the weights.

    Args:
        goal_dim: width of a vector goal, or 0 for a latent goal frame.
        proprio_dim: width of the proprio; its statistics are kept even when
            `use_proprio` is off, as older checkpoints expect.
        contact_dim: width of the contact condition, 0 to leave it out.
    """

    def __init__(
        self,
        action_dim: int,
        latent_dim: int,
        num_tokens: int,
        chunk: int = 5,
        goal_dim: int = 0,
        proprio_dim: int = 6,
        use_proprio: bool = True,
        contact_dim: int = 0,
        train_timesteps: int = 100,
        **denoiser_kw,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.chunk = chunk
        self.goal_dim = goal_dim
        self.use_proprio = use_proprio
        self.use_contact = contact_dim > 0
        self.train_timesteps = train_timesteps
        self.net = Denoiser(
            action_dim, latent_dim, num_tokens, chunk, goal_dim=goal_dim,
            proprio_dim=proprio_dim if use_proprio else 0, contact_dim=contact_dim, **denoiser_kw,
        )
        self.register_buffer("alphas_cumprod", cosine_alphas_cumprod(train_timesteps))
        self.register_buffer("action_low", -torch.ones(action_dim))
        self.register_buffer("action_high", torch.ones(action_dim))
        for name, dim in (("proprio", proprio_dim), ("goal", goal_dim), ("contact", contact_dim)):
            if dim:
                self.register_buffer(f"{name}_mean", torch.zeros(dim))
                self.register_buffer(f"{name}_std", torch.ones(dim))

    def set_stats(self, **stats) -> None:
        """Fill every normalisation buffer: `action_low/high` and `<condition>_mean/std`."""
        expected = set(dict(self.named_buffers(recurse=False))) - {"alphas_cumprod"}
        if set(stats) != expected:
            raise ValueError(f"policy needs statistics {sorted(expected)}, got {sorted(stats)}")
        for name, value in stats.items():
            getattr(self, name).copy_(torch.as_tensor(value, dtype=torch.float32))

    # ---- normalisation --------------------------------------------------- #
    def normalize_action(self, action: torch.Tensor) -> torch.Tensor:
        scale = (self.action_high - self.action_low).clamp_min(1e-6)
        return 2 * (action - self.action_low) / scale - 1

    def unnormalize_action(self, action: torch.Tensor) -> torch.Tensor:
        scale = (self.action_high - self.action_low).clamp_min(1e-6)
        return (action + 1) / 2 * scale + self.action_low

    def _conditions(self, goal, proprio, contact) -> tuple:
        """Z-score the vector conditions; a latent goal passes through, unused ones become None."""
        def zscore(name: str, x: torch.Tensor | None) -> torch.Tensor:
            if x is None:
                raise ValueError(f"this policy was trained with {name}; pass it")
            return (x - getattr(self, f"{name}_mean")) / getattr(self, f"{name}_std")

        return (
            zscore("goal", goal) if self.goal_dim else goal,
            zscore("proprio", proprio) if self.use_proprio else None,
            zscore("contact", contact) if self.use_contact else None,
        )

    # ---- training -------------------------------------------------------- #
    def loss(
        self,
        actions: torch.Tensor,
        z_cur: torch.Tensor,
        goal: torch.Tensor,
        proprio: torch.Tensor | None = None,
        contact: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Epsilon-prediction MSE at a random noise level.

        actions: (B, H, A) raw env actions, z_cur: (B, P, D), goal: (B, P, D) latent
        or (B, goal_dim) raw, proprio: (B, proprio_dim) raw, contact: (B, 1) raw.
        """
        x0 = self.normalize_action(actions)
        b = x0.size(0)
        k = torch.randint(0, self.train_timesteps, (b,), device=x0.device)
        noise = torch.randn_like(x0)
        a = self.alphas_cumprod[k].view(b, 1, 1)
        noisy = a.sqrt() * x0 + (1 - a).sqrt() * noise
        pred = self.net(noisy, k, z_cur, *self._conditions(goal, proprio, contact))
        return F.mse_loss(pred.float(), noise)

    # ---- inference ------------------------------------------------------- #
    @torch.no_grad()
    def sample(
        self,
        z_cur: torch.Tensor,
        goal: torch.Tensor,
        proprio: torch.Tensor | None = None,
        contact: torch.Tensor | None = None,
        steps: int = 10,
    ) -> torch.Tensor:
        """Deterministic DDIM (eta = 0) from pure noise -> raw actions (B, H, A)."""
        b, device = z_cur.size(0), z_cur.device
        goal, proprio, contact = self._conditions(goal, proprio, contact)
        x = torch.randn(b, self.chunk, self.action_dim, device=device)
        ks = torch.linspace(self.train_timesteps - 1, 0, steps, device=device).round().long()
        for i, k in enumerate(ks):
            a = self.alphas_cumprod[k]
            a_prev = self.alphas_cumprod[ks[i + 1]] if i + 1 < steps else torch.ones((), device=device)
            noise = self.net(x, k.expand(b), z_cur, goal, proprio, contact)
            x0 = ((x - (1 - a).sqrt() * noise) / a.sqrt()).clamp(-1, 1)
            x = a_prev.sqrt() * x0 + (1 - a_prev).sqrt() * noise
        return self.unnormalize_action(x)


def build_policy(cfg, action_dim: int, num_tokens: int) -> DiffusionPolicy:
    """Assemble the policy from `cfg.policy`; the latent width is the world model's."""
    pcfg = cfg.policy
    return DiffusionPolicy(
        action_dim=action_dim,
        latent_dim=cfg.model.embed_dim,
        num_tokens=num_tokens,
        chunk=pcfg.chunk,
        goal_dim=GOAL_DIMS[pcfg.goal],
        proprio_dim=PROPRIO_DIMS[pcfg.proprio],
        use_proprio=pcfg.use_proprio,
        contact_dim=1 if pcfg.use_contact else 0,
        train_timesteps=pcfg.train_timesteps,
        width=pcfg.width,
        depth=pcfg.depth,
        heads=pcfg.heads,
        cond_layers=pcfg.cond_layers,
        dropout=pcfg.dropout,
    )


# --------------------------------------------------------------------------- #
#  Planner: the eval.py side
# --------------------------------------------------------------------------- #
class DiffusionPlanner:
    """Act by sampling action chunks from the policy; same interface as `MPCPlanner`.

    Every `policy.chunk` env steps the current frame is encoded, a chunk is
    sampled towards the goal (latent or cube target, as the policy was trained)
    and queued.  With `plan.diffusion_samples` S > 1, S chunks are sampled per
    env and the world model rolls each one out; the chunk whose predicted
    latent lands closest to the goal latent is executed.

    Args:
        policy: a trained `DiffusionPolicy` in eval mode.
        model: the frozen `LeWM` the policy was trained on.
        cfg: the full config (uses `plan`, `data`, `device`).
        normalizer: z-score statistics of the world model's training data.
        image_transform: pixel preprocessing, identical to training.
        action_space_bounds: `(low, high)` of the raw env action space.
    """

    def __init__(self, policy, model, cfg, normalizer, image_transform=None, action_space_bounds=None):
        self.policy = policy
        self.model = model
        self.cfg = cfg
        self.plan = cfg.plan
        self.normalizer = normalizer
        self.image_transform = image_transform
        self.device = torch.device(cfg.device)
        self.obs_key = cfg.data.obs_key
        self.action_dim = policy.action_dim
        self.bounds = action_space_bounds

        self.num_envs = 0
        self.queue: list[deque] = []
        self.goal_emb: torch.Tensor | None = None
        self.goal_pos: torch.Tensor | None = None
        self.last_cost: np.ndarray | None = None

    def reset(self, num_envs: int) -> None:
        self.num_envs = num_envs
        self.queue = [deque() for _ in range(num_envs)]
        self.goal_emb = None
        self.goal_pos = None
        self.last_cost = None

    @torch.no_grad()
    def set_goal(self, goal_obs, goal_pos=None) -> None:
        """goal_obs: goal observations (N, ...); goal_pos: (N, 3) cube targets in metres.

        The goal latent is kept for every policy, since `plan.diffusion_samples > 1`
        ranks chunks against it; `goal_pos` is required by a `policy.goal=cube_xyz` policy.
        """
        obs = self._to_model_input(goal_obs)
        self.goal_emb = self.model.encode(obs.unsqueeze(1))[:, 0]  # (N, P, D)
        if self.policy.goal_dim:
            if goal_pos is None:
                raise ValueError("this policy is conditioned on the cube's target position; pass goal_pos")
            self.goal_pos = torch.as_tensor(np.asarray(goal_pos, np.float32), device=self.device)

    @torch.no_grad()
    def act(self, obs: dict, active: np.ndarray | None = None) -> np.ndarray:
        """obs: the env's observation dict (needs `data.obs_key`, `proprio`, and
        `contact` for a policy that uses it) -> (N, A)."""
        if self.goal_emb is None:
            raise RuntimeError("call set_goal() before act()")
        active = np.ones(self.num_envs, dtype=bool) if active is None else np.asarray(active, bool)

        replan = [i for i in range(self.num_envs) if active[i] and not self.queue[i]]
        if replan:
            self._replan(replan, obs)

        actions = np.zeros((self.num_envs, self.action_dim), dtype=np.float32)
        for i in range(self.num_envs):
            if active[i] and self.queue[i]:
                actions[i] = self.queue[i].popleft()
        return actions

    def _replan(self, envs: list[int], obs: dict) -> None:
        n, s = len(envs), self.plan.diffusion_samples
        idx = torch.as_tensor(envs, dtype=torch.long, device=self.device)
        z_cur = self.model.encode(self._to_model_input(obs[self.obs_key][envs]).unsqueeze(1))  # (n, 1, P, D)
        z_goal = self.goal_emb[idx]  # (n, P, D)
        goal = self.goal_pos[idx] if self.policy.goal_dim else z_goal
        proprio = torch.as_tensor(obs["proprio"][envs], dtype=torch.float32, device=self.device)
        contact = (
            torch.as_tensor(obs["contact"][envs], dtype=torch.float32, device=self.device)
            if self.policy.use_contact
            else None
        )

        def rep(x):  # one copy per sample
            return None if x is None else x.repeat_interleave(s, 0)

        chunks = self.policy.sample(
            rep(z_cur[:, 0]), rep(goal), rep(proprio), rep(contact), steps=self.plan.ddim_steps,
        ).view(n, s, self.policy.chunk, self.action_dim)
        if self.bounds is not None:
            low, high = (torch.as_tensor(b, dtype=chunks.dtype, device=chunks.device) for b in self.bounds)
            chunks = torch.maximum(torch.minimum(chunks, high), low)  # score what will be executed

        if s > 1:
            # The world model sees z-scored actions grouped into frameskip blocks.
            fs = self.cfg.data.frameskip
            candidates = self.normalizer.normalize("action", chunks)
            candidates = candidates.reshape(n, s, self.policy.chunk // fs, self.action_dim * fs)
            cost = self.model.plan_cost(z_cur, z_goal, candidates, self.plan.chunk_size)  # (n, s)
            best = cost.argmin(dim=1)
            self.last_cost = np.full(self.num_envs, np.nan, dtype=np.float32)
            self.last_cost[envs] = cost.min(dim=1).values.float().cpu().numpy()
        else:
            best = torch.zeros(n, dtype=torch.long, device=chunks.device)

        plan = chunks[torch.arange(n, device=chunks.device), best].float().cpu().numpy()  # (n, H, A)
        for row, env_i in enumerate(envs):
            self.queue[env_i].extend(plan[row])

    def _to_model_input(self, obs) -> torch.Tensor:
        return to_model_input(obs, self.obs_key, self.image_transform, self.normalizer, self.device)
