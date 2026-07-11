from ..config import ExperimentConfig
from blockA_experiments import run_blockA as _run_blockA


def run_blockA(cfg: ExperimentConfig):
    """
    Thin wrapper so other blocks can call Block A runner via geomctrl.
    """
    return _run_blockA(cfg)


__all__ = ["run_blockA"]

