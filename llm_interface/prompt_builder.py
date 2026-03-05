# llm_interface/prompt_builder.py
#
# Builds the LLM system prompt from the predicate catalogue, scene library,
# and few-shot JSON examples.

import json
import os
from llm_interface.predicate_catalogue import CATALOGUE, WEIGHT_RANGE
from llm_interface.scene_library import SCENE_LIBRARY

# ── Few-shot example JSONs (relative to workspace root) ──────────────────── #
_EXAMPLE_FILES = [
    "spec/exp1_task.json",
    "spec/exp2_task.json",
    "spec/exp3a_task.json",
]

_WORKSPACE_ROOT = os.path.join(os.path.dirname(__file__), "..")


def _load_example(rel_path: str) -> str:
    """Load JSON example file and return as indented string."""
    full = os.path.join(_WORKSPACE_ROOT, rel_path)
    try:
        with open(full) as f:
            return json.dumps(json.load(f), indent=2)
    except FileNotFoundError:
        return f"[example file {rel_path} not found]"


def _build_catalogue_section() -> str:
    lines = ["=== PREDICATE CATALOGUE ===\n"]
    lines.append(
        "These are the ONLY predicates you may use. "
        "Do NOT invent new predicate names.\n"
    )
    for name, cat in CATALOGUE.items():
        lines.append(f"  {name}")
        lines.append(f"    allowed_modalities : {cat['allowed_modalities']}")
        lines.append(f"    allowed_operators  : {cat['allowed_operators']}")
        lines.append("    parameters:")
        for pname, pdef in cat["params"].items():
            info = f"      {pname}: type={pdef['type']}"
            if "default" in pdef:
                info += f", default={pdef['default']}"
            if "min" in pdef and "max" in pdef:
                info += f", range=[{pdef['min']}, {pdef['max']}]"
            lines.append(info)
        if cat.get("has_hard_strength"):
            lines.append(
                f"    hard_strength      : default={cat['hard_strength_default']}, "
                f"range={cat['hard_strength_range']}  (only for HARD modality)"
            )
            lines.append(
                f"    hard_infl_factor   : default={cat['hard_infl_factor_default']}, "
                f"range={cat['hard_infl_factor_range']}  (only for HARD modality)"
            )
        lines.append("")
    return "\n".join(lines)


def _build_weight_section() -> str:
    return """\
=== WEIGHT RULES ===

PREFER  clauses: weight IS the optimizer cost multiplier.
  - Range: [1.0, 20.0].  DO NOT exceed 20.
  - Critical preference (must reach goal): 8–15
  - Strong preference (avoid human comfort): 10–15
  - Soft style (minor preference): 1–5

REQUIRE clauses: weight field is documentation only (ignored by optimizer).
  - Set weight = 10.0 as a convention.

HARD    clauses: weight field is documentation only (ignored by optimizer).
  - Set weight = 10.0 as a convention.

IMPORTANT: Never output weight > 20.0 for any clause.
Weights of 100, 500, 1000, 2000 are WRONG and will be rejected.
"""


def _build_scene_library_section() -> str:
    lines = ["=== SCENE ENTITY LIBRARY ===\n"]
    lines.append(
        "Use these entity names to look up physical parameters. "
        "Positions come from the user description or camera.\n"
    )
    for name, params in SCENE_LIBRARY.items():
        lines.append(f"  {name}:")
        for k, v in params.items():
            lines.append(f"    {k}: {v}")
        lines.append("")
    return "\n".join(lines)


def _build_modality_rules_section() -> str:
    return """\
=== MODALITY RULES (STRICT) ===

HARD    — use for physical safety constraints that must NEVER be violated.
          Triggers DMP repulsion + geometric projection (guaranteed safe).
          Examples: collision avoidance, human body exclusion.

REQUIRE — use for task-completion constraints (optimizer penalises violation).
          Examples: reach goal, stay below velocity limit, orientation limit.

PREFER  — use for soft preferences (optimizer trades off against other costs).
          Examples: comfort distance from human, smooth path style.

MODALITY IS NOT YOUR CHOICE FOR THESE — fixed rules:
  HumanBodyExclusion    → MUST be HARD
  HumanComfortDistance  → MUST be PREFER
  VelocityLimit         → MUST be REQUIRE
  AngularVelocityLimit  → MUST be REQUIRE
  ZeroVelocity          → MUST be REQUIRE
  OrientationLimit      → MUST be REQUIRE
  HoldAtWaypoint        → MUST be REQUIRE
"""


def _build_output_format_section() -> str:
    return """\
=== OUTPUT FORMAT ===

Output ONLY a single valid JSON object. No markdown, no explanation, no
code fences. The JSON must have these top-level keys:

  "horizon_sec"  : float  — total task duration in seconds
  "phases"       : list   — list of DMP phase dicts (see examples)
  "clauses"      : list   — list of clause dicts
  "bindings"     : dict   — "PredicateName.param_name": value

Each clause dict must have:
  "type"     : operator string
  "predicate": predicate name from catalogue
  "weight"   : float in [1.0, 20.0]
  "modality" : "HARD", "REQUIRE", or "PREFER"

For HARD obstacle/human clauses, also include:
  "hard_strength"    : float
  "hard_infl_factor" : float

For *_during operators, also include:
  "time_window": [t_start, t_end]

The "bindings" dict uses keys of the form "PredicateName.param_name".
"""


def build_system_prompt(include_examples: bool = True) -> str:
    """Build the full system prompt for the LLM."""
    parts = [
        "You are a robotic task specification compiler. "
        "Given a natural language description of a manipulation task, "
        "you output a JSON task specification for the CLAVIC system.\n",
        _build_modality_rules_section(),
        _build_weight_section(),
        _build_catalogue_section(),
        _build_scene_library_section(),
        _build_output_format_section(),
    ]

    if include_examples:
        parts.append("=== FEW-SHOT EXAMPLES ===\n")
        for path in _EXAMPLE_FILES:
            parts.append(f"--- Example ({path}) ---")
            parts.append(_load_example(path))
            parts.append("")

    return "\n".join(parts)
