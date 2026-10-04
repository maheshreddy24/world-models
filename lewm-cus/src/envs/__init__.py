"""Simulation environments used for closed-loop evaluation.

Each wrapper drags in its own simulator stack — `stable_worldmodel` for the cube
task, `ogbench` for the scene one — so they are imported on first use rather
than both at package import.
"""

__all__ = ["BallVecEnv", "CubeVecEnv", "CubeDoubleVecEnv", "SceneVecEnv"]

_MODULES = {"BallVecEnv": ".ball", "CubeVecEnv": ".cube", "CubeDoubleVecEnv": ".cube_double", "SceneVecEnv": ".scene"}


def __getattr__(name: str):
    if name not in _MODULES:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module

    return getattr(import_module(_MODULES[name], __name__), name)


def __dir__():
    return sorted(__all__)
