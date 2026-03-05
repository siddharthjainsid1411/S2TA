# llm_interface/validator.py
#
# Two-pass validation and clamping of LLM-generated task spec dicts.
#
# Pass 1 — structural checks: unknown predicates, illegal modalities, missing
#           required bindings. Returns a list of errors to feed back to the LLM.
# Pass 2 — silent numeric clamping: weights and parameter values are clamped to
#           catalogue-defined ranges. Issues are logged as warnings only.

import json
import logging
from llm_interface.predicate_catalogue import CATALOGUE, WEIGHT_RANGE

logger = logging.getLogger(__name__)

VALID_OPERATORS = {
    "always", "eventually",
    "always_during", "eventually_during",
    "until",
}
VALID_MODALITIES = {"HARD", "REQUIRE", "PREFER"}


# ─────────────────────────────────────────────────────────────────────────── #
#  Public entry point
# ─────────────────────────────────────────────────────────────────────────── #

def validate_and_clamp(spec_dict: dict) -> tuple[dict, list[str], list[str]]:
    """
    Validate and clamp a raw LLM-produced spec dict.

    Returns
    -------
    (fixed_dict, errors, warnings)
        errors    : hard structural errors requiring LLM retry
        warnings  : values that were silently clamped
        fixed_dict: corrected copy of the input
    """
    spec = json.loads(json.dumps(spec_dict))
    errors   = []
    warnings = []

    # Top-level structure
    if "horizon_sec" not in spec:
        errors.append("Missing 'horizon_sec' at top level.")
    else:
        hs = spec["horizon_sec"]
        if not isinstance(hs, (int, float)) or hs <= 0:
            errors.append(f"'horizon_sec' must be a positive number, got {hs!r}.")
        elif hs < 1.0 or hs > 60.0:
            clamped = max(1.0, min(60.0, float(hs)))
            warnings.append(f"horizon_sec={hs} clamped to {clamped}.")
            spec["horizon_sec"] = clamped

    if "clauses" not in spec or not isinstance(spec["clauses"], list):
        errors.append("Missing or empty 'clauses' list.")
        return spec, errors, warnings   # can't continue without clauses

    if "bindings" not in spec or not isinstance(spec["bindings"], dict):
        errors.append("Missing 'bindings' dict.")
        return spec, errors, warnings

    bindings = spec["bindings"]

    # Per-clause validation
    for idx, clause in enumerate(spec["clauses"]):
        tag = f"clause[{idx}]"

        # Required fields present?
        for field in ("type", "predicate", "weight", "modality"):
            if field not in clause:
                errors.append(f"{tag}: missing field '{field}'.")

        if len(errors) > 0:
            continue  # skip further checks on malformed clause

        operator  = clause["type"]
        predicate = clause["predicate"]
        modality  = clause["modality"]
        weight    = clause["weight"]

        if operator not in VALID_OPERATORS:
            errors.append(
                f"{tag} ({predicate}): unknown operator '{operator}'. "
                f"Valid: {sorted(VALID_OPERATORS)}"
            )

        if modality not in VALID_MODALITIES:
            errors.append(
                f"{tag} ({predicate}): unknown modality '{modality}'. "
                f"Valid: HARD, REQUIRE, PREFER"
            )

        if predicate not in CATALOGUE:
            errors.append(
                f"{tag}: unknown predicate '{predicate}'. "
                f"Valid predicates: {sorted(CATALOGUE.keys())}"
            )
            continue

        cat = CATALOGUE[predicate]

        if modality not in cat["allowed_modalities"]:
            errors.append(
                f"{tag} ({predicate}): modality '{modality}' not allowed. "
                f"This predicate only accepts: {cat['allowed_modalities']}"
            )

        if operator in VALID_OPERATORS and operator not in cat["allowed_operators"]:
            errors.append(
                f"{tag} ({predicate}): operator '{operator}' not allowed. "
                f"This predicate only accepts: {cat['allowed_operators']}"
            )

        if operator in ("always_during", "eventually_during"):
            if "time_window" not in clause:
                errors.append(
                    f"{tag} ({predicate}): operator '{operator}' requires "
                    f"'time_window': [t_start, t_end]."
                )

        if modality in WEIGHT_RANGE and isinstance(weight, (int, float)):
            lo, hi = WEIGHT_RANGE[modality]
            if weight < lo or weight > hi:
                clamped_w = max(lo, min(hi, float(weight)))
                warnings.append(
                    f"{tag} ({predicate}): weight={weight} out of range "
                    f"[{lo},{hi}] for {modality} → clamped to {clamped_w}."
                )
                clause["weight"] = clamped_w

        for param_name in cat["params"].keys():
            binding_key = f"{predicate}.{param_name}"
            if binding_key not in bindings:
                param_def = cat["params"][param_name]
                if "default" in param_def:
                    warnings.append(
                        f"{tag} ({predicate}): binding '{binding_key}' not "
                        f"found; using default {param_def['default']}."
                    )
                    bindings[binding_key] = param_def["default"]
                else:
                    errors.append(
                        f"{tag} ({predicate}): required binding '{binding_key}' "
                        f"is missing from 'bindings' dict."
                    )

        for param_name, param_def in cat["params"].items():
            binding_key = f"{predicate}.{param_name}"
            val = bindings.get(binding_key)
            if val is None:
                continue
            if param_def["type"] == "float" and isinstance(val, (int, float)):
                lo = param_def.get("min")
                hi = param_def.get("max")
                if lo is not None and hi is not None:
                    if val < lo or val > hi:
                        clamped_v = max(lo, min(hi, float(val)))
                        warnings.append(
                            f"binding '{binding_key}'={val} out of range "
                            f"[{lo},{hi}] → clamped to {clamped_v}."
                        )
                        bindings[binding_key] = clamped_v

        if cat.get("has_hard_strength"):
            hs_lo, hs_hi = cat.get("hard_strength_range", (0.01, 1.0))
            hif_lo, hif_hi = cat.get("hard_infl_factor_range", (1.0, 6.0))

            if "hard_strength" in clause:
                v = clause["hard_strength"]
                if isinstance(v, (int, float)) and (v < hs_lo or v > hs_hi):
                    cv = max(hs_lo, min(hs_hi, float(v)))
                    warnings.append(
                        f"{tag} ({predicate}): hard_strength={v} → clamped to {cv}."
                    )
                    clause["hard_strength"] = cv
            else:
                clause["hard_strength"] = cat["hard_strength_default"]

            if "hard_infl_factor" in clause:
                v = clause["hard_infl_factor"]
                if isinstance(v, (int, float)) and (v < hif_lo or v > hif_hi):
                    cv = max(hif_lo, min(hif_hi, float(v)))
                    warnings.append(
                        f"{tag} ({predicate}): hard_infl_factor={v} → clamped to {cv}."
                    )
                    clause["hard_infl_factor"] = cv
            else:
                clause["hard_infl_factor"] = cat["hard_infl_factor_default"]

    if warnings:
        for w in warnings:
            logger.warning("VALIDATOR CLAMP: %s", w)
    if errors:
        for e in errors:
            logger.error("VALIDATOR ERROR: %s", e)

    return spec, errors, warnings