# llm_interface/predicate_catalogue.py
#
# Single source of truth for every predicate the LLM may use:
#   - allowed modalities and temporal operators per predicate
#   - parameter names, types, and admissible ranges
#   - weight clamp ranges per modality
#
# The validator uses this to hard-reject or silently clamp any LLM output
# before it reaches the optimizer.

# Weight clamp ranges per modality.
# PREFER weights are the actual optimizer cost multipliers.
# REQUIRE and HARD weights are documentation only (compiler uses fixed 500),
# but are still validated to catch out-of-range values.
WEIGHT_RANGE = {
    "PREFER":  (1.0, 20.0),
    "REQUIRE": (1.0, 20.0),
    "HARD":    (1.0, 20.0),
}

# Per-predicate catalogue entry schema:
#   allowed_modalities : modality strings the LLM may assign
#   allowed_operators  : STL operator strings
#   params             : param_name → {type, default?, min?, max?}
#                        type is one of: float, list3, list4
#   has_hard_strength  : True if hard_strength / hard_infl_factor apply
CATALOGUE = {

    "AtGoal": {
        "allowed_modalities": ["REQUIRE", "PREFER"],
        "allowed_operators":  ["eventually", "eventually_during"],
        "params": {
            "waypoint":   {"type": "list3"},
            "tolerance":  {"type": "float", "default": 0.03, "min": 0.01, "max": 0.15},
        },
        "has_hard_strength": False,
    },

    "AtWaypoint": {
        "allowed_modalities": ["REQUIRE", "PREFER"],
        "allowed_operators":  ["eventually", "eventually_during"],
        "params": {
            "waypoint":   {"type": "list3"},
            "tolerance":  {"type": "float", "default": 0.03, "min": 0.01, "max": 0.15},
        },
        "has_hard_strength": False,
    },

    "HoldAtWaypoint": {
        "allowed_modalities": ["REQUIRE"],
        "allowed_operators":  ["always_during"],
        "params": {
            "waypoint":         {"type": "list3"},
            "tolerance":        {"type": "float", "default": 0.03, "min": 0.01, "max": 0.10},
            "speed_threshold":  {"type": "float", "default": 0.05, "min": 0.01, "max": 0.20},
        },
        "has_hard_strength": False,
    },

    "HumanBodyExclusion": {
        "allowed_modalities": ["HARD"],
        "allowed_operators":  ["always"],
        "params": {
            "human_position": {"type": "list3"},
            "body_radius":    {"type": "float", "default": 0.08, "min": 0.05, "max": 0.20},
        },
        "has_hard_strength": True,
        "hard_strength_default":    0.20,
        "hard_strength_range":      (0.05, 0.40),
        "hard_infl_factor_default": 3.0,
        "hard_infl_factor_range":   (1.5, 5.0),
    },

    "HumanComfortDistance": {
        "allowed_modalities": ["PREFER"],
        "allowed_operators":  ["always"],
        "params": {
            "human_position":      {"type": "list3"},
            "preferred_distance":  {"type": "float", "default": 0.19, "min": 0.10, "max": 0.50},
        },
        "has_hard_strength": False,
    },

    "ObstacleAvoidance": {
        "allowed_modalities": ["HARD", "PREFER"],
        "allowed_operators":  ["always"],
        "params": {
            "obstacle_position": {"type": "list3"},
            "safe_radius":       {"type": "float", "default": 0.10, "min": 0.03, "max": 0.30},
        },
        "has_hard_strength": True,
        "hard_strength_default":    0.05,
        "hard_strength_range":      (0.03, 0.20),
        "hard_infl_factor_default": 2.5,
        "hard_infl_factor_range":   (1.5, 4.0),
    },

    "VelocityLimit": {
        "allowed_modalities": ["REQUIRE"],
        "allowed_operators":  ["always", "always_during"],
        "params": {
            "vmax": {"type": "float", "default": 0.5, "min": 0.05, "max": 2.0},
        },
        "has_hard_strength": False,
    },

    "AngularVelocityLimit": {
        "allowed_modalities": ["REQUIRE"],
        "allowed_operators":  ["always", "always_during"],
        "params": {
            "omega_max": {"type": "float", "default": 1.0, "min": 0.1, "max": 5.0},
        },
        "has_hard_strength": False,
    },

    "ZeroVelocity": {
        "allowed_modalities": ["REQUIRE"],
        "allowed_operators":  ["always_during"],
        "params": {
            "speed_threshold": {"type": "float", "default": 0.05, "min": 0.01, "max": 0.20},
        },
        "has_hard_strength": False,
    },

    "OrientationLimit": {
        "allowed_modalities": ["REQUIRE"],
        "allowed_operators":  ["always", "always_during"],
        "params": {
            "q_ref":         {"type": "list4"},
            "max_angle_rad": {"type": "float", "default": 0.3, "min": 0.05, "max": 1.57},
        },
        "has_hard_strength": False,
    },

    "OrientationAtTarget": {
        "allowed_modalities": ["REQUIRE", "PREFER"],
        "allowed_operators":  ["eventually", "eventually_during"],
        "params": {
            "q_target":       {"type": "list4"},
            "tolerance_rad":  {"type": "float", "default": 0.1, "min": 0.03, "max": 0.50},
        },
        "has_hard_strength": False,
    },

    "OrientationHold": {
        "allowed_modalities": ["REQUIRE"],
        "allowed_operators":  ["always_during"],
        "params": {
            "q_target":       {"type": "list4"},
            "tolerance_rad":  {"type": "float", "default": 0.1, "min": 0.03, "max": 0.50},
            "omega_max":      {"type": "float", "default": 0.05, "min": 0.01, "max": 0.50},
        },
        "has_hard_strength": False,
    },

    "DirectionalStiffnessNearHuman": {
        "allowed_modalities": ["REQUIRE", "PREFER"],
        "allowed_operators":  ["always"],
        "params": {
            "human_position":     {"type": "list3"},
            "proximity_radius":   {"type": "float", "default": 0.20, "min": 0.05, "max": 0.50},
            "k_max_near_human":   {"type": "float", "default": 80.0, "min": 10.0, "max": 200.0},
        },
        "has_hard_strength": False,
    },
}


def get_predicate_names():
    """Return sorted list of all valid predicate names."""
    return sorted(CATALOGUE.keys())


def get_catalogue_entry(predicate_name):
    """Return catalogue entry or None if predicate unknown."""
    return CATALOGUE.get(predicate_name, None)
