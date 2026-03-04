# spec/taskspec.py

from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple, Optional


@dataclass
class Clause:
    operator: str          # "eventually", "always", "until", "always_during", "eventually_during"
    predicate: str         # name of predicate (or tuple of two names for "until")
    weight: float          # logic weight
    modality: str          # "HARD", "REQUIRE", or "PREFER"
    #   HARD    — 3-layer geometric guarantee:
    #               Layer 1: DMP repulsive forcing inside ODE
    #               Layer 2: post-rollout radial projector (∀t: ||p-c|| ≥ r)
    #               Layer 3: slack penalty in optimizer (same as REQUIRE)
    #             The JSON parser extracts obstacle specs automatically and
    #             policy.setup_hard_obstacles_from_taskspec() wires Layers 1+2.
    #   REQUIRE — optimizer slack penalty only (no geometric guarantee).
    #   PREFER  — weighted soft cost; can trade off against other objectives.
    parameters: Dict[str, Any] = field(default_factory=dict)
    time_window: Optional[Tuple[float, float]] = None   # [t_start, t_end] for *_during ops
    # For HARD clauses only: obstacle spec extracted by json_parser
    hard_obstacle: Optional[Dict[str, Any]] = None


@dataclass
class TaskSpec:
    horizon_sec: float
    clauses: List[Clause]
    auxiliary_weights: Dict[str, float] = field(default_factory=dict)
    # Populated by json_parser from all HARD clauses — fed to set_obstacles()
    hard_obstacle_specs: List[Dict[str, Any]] = field(default_factory=list)