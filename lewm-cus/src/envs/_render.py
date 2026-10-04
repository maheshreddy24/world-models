"""Shutdown helper shared by the MuJoCo env wrappers."""

from __future__ import annotations


def close_env(env) -> None:
    """Close a gym env and free the `mujoco.Renderer` OGBench leaves open.

    OGBench's envs never close their renderer, so its EGL context outlives
    `env.close()`. MuJoCo terminates the EGL display in an atexit hook, and the
    context's `__del__` then fails at interpreter exit with EGL_NOT_INITIALIZED.
    Harmless, but it buries the script's output in tracebacks.
    """
    base = env.unwrapped
    renderer = getattr(base, "_renderer", None)
    if renderer is not None:
        renderer.close()
        base._renderer = None
    env.close()
