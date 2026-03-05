# llm_interface/scene_library.py
#
# Pre-calibrated physical parameters for known scene entities.
# Positions are defaults and can be updated at runtime by a camera or
# perception pipeline via update_entity_position().

SCENE_LIBRARY = {

    "human": {
        "body_radius":      0.08,
        "comfort_radius":   0.19,
        "hard_strength":    0.20,
        "hard_infl_factor": 3.0,
        "position":         [0.30, 0.30, 0.30],
    },

    "laptop": {
        "safe_radius":      0.15,
        "hard_strength":    0.08,
        "hard_infl_factor": 2.5,
        "position":         [0.40, 0.20, 0.25],
    },

    "obstacle": {
        "safe_radius":      0.10,
        "hard_strength":    0.05,
        "hard_infl_factor": 2.0,
        "position":         [0.40, 0.30, 0.30],
    },

    "fragile": {
        "safe_radius":      0.12,
        "hard_strength":    0.15,
        "hard_infl_factor": 3.0,
        "position":         [0.35, 0.25, 0.30],
    },

    "wall": {
        "safe_radius":      0.08,
        "hard_strength":    0.10,
        "hard_infl_factor": 2.0,
        "position":         [0.20, 0.40, 0.30],
    },
}


def get_entity(name: str) -> dict:
    """Return a copy of the parameter dict for a named scene entity."""
    if name not in SCENE_LIBRARY:
        raise KeyError(
            f"Entity '{name}' not in scene library. "
            f"Available: {list(SCENE_LIBRARY.keys())}"
        )
    return dict(SCENE_LIBRARY[name])


def list_entities() -> list:
    """Return names of all pre-defined scene entities."""
    return list(SCENE_LIBRARY.keys())


def update_entity_position(name: str, position: list):
    """Update the position of a scene entity (e.g. from a perception pipeline)."""
    if name not in SCENE_LIBRARY:
        raise KeyError(f"Entity '{name}' not in scene library.")
    SCENE_LIBRARY[name]["position"] = list(position)
