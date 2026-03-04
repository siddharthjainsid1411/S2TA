import json
import numpy as np
from spec.taskspec import TaskSpec, Clause

# Predicates that represent spherical obstacles — used to auto-extract
# obstacle specs when modality="HARD".
_OBSTACLE_PREDICATES = {
    "ObstacleAvoidance":  ("obstacle_position", "safe_radius"),
    "HumanBodyExclusion": ("human_position",    "body_radius"),
}

def load_taskspec_from_json(path):

    with open(path, "r") as f:
        data = json.load(f)

    horizon_sec = data["horizon_sec"]
    bindings = data.get("bindings", {})
    phases = data.get("phases", None)

    clauses = []
    hard_obstacle_specs = []

    for c in data["clauses"]:

        operator = c["type"]
        weight   = c["weight"]
        modality = c["modality"]   # "HARD", "REQUIRE", or "PREFER"

        time_window = None
        if "time_window" in c:
            time_window = tuple(c["time_window"])

        if operator in ["always", "eventually", "always_during", "eventually_during"]:

            predicate = c["predicate"]
            parameters = extract_parameters(predicate, bindings)

            # ── HARD: extract obstacle spec for Layers 1+2 ──────────────
            hard_obstacle = None
            if modality == "HARD" and predicate in _OBSTACLE_PREDICATES:
                center_key, radius_key = _OBSTACLE_PREDICATES[predicate]
                center = parameters.get(center_key)
                radius = parameters.get(radius_key)
                if center is not None and radius is not None:
                    center_list = center.tolist() if hasattr(center, "tolist") else list(center)
                    hard_obstacle = {
                        "center":      center_list,
                        "radius":      float(radius),
                        "avoidance":   "HARD",
                        "strength":    float(c.get("hard_strength",    0.05)),
                        "infl_factor": float(c.get("hard_infl_factor", 2.5)),
                    }
                    hard_obstacle_specs.append(hard_obstacle)

            clause = Clause(
                operator=operator,
                predicate=predicate,
                weight=weight,
                modality=modality,
                parameters=parameters,
                time_window=time_window,
                hard_obstacle=hard_obstacle,
            )

        elif operator == "until":

            left  = c["left"]
            right = c["right"]
            parameters = {
                "left_params":  extract_parameters(left,  bindings),
                "right_params": extract_parameters(right, bindings),
            }
            clause = Clause(
                operator=operator,
                predicate=(left, right),
                weight=weight,
                modality=modality,
                parameters=parameters,
                time_window=time_window,
            )

        else:
            raise ValueError(f"Unsupported operator: {operator}")

        clauses.append(clause)

    ts = TaskSpec(
        horizon_sec=horizon_sec,
        clauses=clauses,
        hard_obstacle_specs=hard_obstacle_specs,
    )
    ts.phases = phases
    return ts


def extract_parameters(predicate_name, bindings):

    params = {}
    for key, value in bindings.items():
        if key.startswith(predicate_name + "."):
            param_name = key.split(".")[1]
            if isinstance(value, list):
                value = np.array(value, dtype=float)
            params[param_name] = value
    return params
