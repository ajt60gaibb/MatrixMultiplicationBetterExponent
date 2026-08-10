#!/usr/bin/env python3
"""Turn optimized CW^4 region rows into a feasible standalone certificate.

The optimized rows are installed in a fresh verifier model.  Every auxiliary
retention/minimum variable is then reset below its newly inferred limiting
value, and omega is reset above the resulting Schonhage requirement.  The
saved MAT file is reloaded into another fresh model for the final report.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from scipy.io import savemat

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path[:0] = [str(ROOT / "verification"), str(HERE)]

from python_verifier import Model, TermInfo  # noqa: E402
from global_stage_opt import CachedGlobalObjective  # noqa: E402
import p4_lower_region_polish as p4_module  # noqa: E402
from p4_lower_region_polish import P4RegionPolisher, masked_softmax  # noqa: E402


def install_rows(model: Model, rows: np.ndarray) -> list[int]:
    indices = []
    local = 0
    for term_index, term in enumerate(model.terms[3]):
        if not isinstance(term, TermInfo):
            continue
        row = np.asarray(rows[local], dtype=float)
        model.pm.x[term.region_prop_id.sl] = row
        indices.append(term_index)
        local += 1
    if local != len(rows):
        raise ValueError(f"candidate has {len(rows)} rows, model consumed {local}")
    return indices


def reset_auxiliaries(model: Model, inferred: dict, margin: float, omega_margin: float) -> None:
    # L3 retentions for each orientation.
    for region in range(6):
        value = float(inferred["inferred_comp"][region][3]) - margin
        model.pm.x[model.num_retain_comp_id[region][3].sl] = max(value, 0.0)
    # The symmetric L2 retention is stored only in region zero; the other five
    # variables are constrained equal to zero.
    value2 = float(inferred["inferred_comp"][0][2]) - margin
    model.pm.x[model.num_retain_comp_id[0][2].sl] = max(value2, 0.0)
    for region in range(1, 6):
        model.pm.x[model.num_retain_comp_id[region][2].sl] = 0.0
    for region in range(6):
        value = float(inferred["inferred_global"][region]) - margin
        model.pm.x[model.num_retain_glob_id[region].sl] = max(value, 0.0)
    single = float(inferred["inferred_single"]) - margin
    model.pm.x[model.single_mat_size_id.sl] = max(single, margin)

    retained = sum(float(model.pm.get(group)[0]) for group in model.num_retain_glob_id)
    for level in range(2, model.max_level + 1):
        retained += sum(
            float(model.pm.get(model.num_retain_comp_id[region][level])[0])
            for region in range(6)
        )
    required = (float(inferred["target"]) - retained) / float(
        model.pm.get(model.single_mat_size_id)[0]
    )
    model.pm.x[model.omega_id.sl] = required + omega_margin


def ranked_constraints(result: dict, count: int = 12) -> dict:
    inequalities = sorted(result["inequalities"], key=lambda item: item[1], reverse=True)[:count]
    equalities = result["equalities"] + result["linear_equalities"]
    equalities = sorted(
        equalities,
        key=lambda item: float(np.max(np.abs(item[1]))),
        reverse=True,
    )[:count]
    return {
        "inequalities": [[label, float(value)] for label, value in inequalities],
        "equalities": [
            [label, float(np.max(np.abs(values)))] for label, values in equalities
        ],
    }


def gradient_audit(certificate: Path, rows: np.ndarray, seed: int) -> dict:
    """Compare the analytic row gradient with fresh-verifier finite differences."""
    model = Model(5, 3, 1)
    model.load(certificate)
    lower = CachedGlobalObjective(model)
    program = P4RegionPolisher(model, lower)
    if rows.shape != program.pi0.shape:
        raise ValueError("row shape mismatch in gradient audit")
    logits = np.log(np.maximum(rows, 1e-300))
    rng = np.random.default_rng(seed)
    direction = rng.standard_normal(rows.shape)
    direction[~program.mask] = 0.0
    direction /= np.linalg.norm(direction)
    def fresh_value(candidate_rows: np.ndarray) -> float:
        fresh = Model(5, 3, 1)
        fresh.load(certificate)
        install_rows(fresh, candidate_rows)
        return float(fresh.evaluate()["tightened_omega"])

    # The exact candidate has matrix-coordinate gaps down to 3.3e-12.  Use a
    # genuinely unique argmin for the analytic derivative and an epsilon
    # ladder extending below the branch-switching scale.  The optimizer's
    # 3e-11 active averaging is useful numerically but is not the derivative
    # of this slightly unbalanced stored point.
    original_softmin = p4_module.softmin

    def onehot_min(values: np.ndarray, _temperature: float):
        values = np.asarray(values, dtype=float)
        weights = np.zeros_like(values)
        weights[int(np.argmin(values))] = 1.0
        return float(values.min()), weights

    p4_module.softmin = onehot_min
    try:
        _, exact_row_gradient = program.value_gradient(rows, 0.0)
    finally:
        p4_module.softmin = original_softmin
    exact_logit_gradient = rows * (
        exact_row_gradient - np.sum(rows * exact_row_gradient, axis=1, keepdims=True)
    )
    exact_analytic = float(np.sum(exact_logit_gradient * direction))
    exact_ladder = []
    direct_crosscheck = []
    for epsilon in (3e-4, 1e-4, 3e-5, 1e-5, 3e-6, 1e-6, 3e-7, 1e-7):
        plus = masked_softmax(logits + epsilon * direction, program.mask)
        minus = masked_softmax(logits - epsilon * direction, program.mask)
        fresh_plus = fresh_value(plus)
        fresh_minus = fresh_value(minus)
        numerical = (fresh_plus - fresh_minus) / (2 * epsilon)
        exact_ladder.append(
            {
                "epsilon": epsilon,
                "analytic": exact_analytic,
                "numerical": numerical,
                "absolute_error": abs(numerical - exact_analytic),
            }
        )
        direct_crosscheck.append(
            {
                "epsilon": epsilon,
                "plus_difference": fresh_plus - program.forward(plus),
                "minus_difference": fresh_minus - program.forward(minus),
            }
        )

    smooth_temperature = 1e-6
    _, smooth_row_gradient = program.value_gradient(rows, smooth_temperature)
    smooth_logit_gradient = rows * (
        smooth_row_gradient - np.sum(rows * smooth_row_gradient, axis=1, keepdims=True)
    )
    smooth_analytic = float(np.sum(smooth_logit_gradient * direction))
    smooth_ladder = []
    for epsilon in (3e-4, 1e-4, 3e-5, 1e-5):
        plus = masked_softmax(logits + epsilon * direction, program.mask)
        minus = masked_softmax(logits - epsilon * direction, program.mask)
        numerical = (
            program.value_gradient(plus, smooth_temperature)[0]
            - program.value_gradient(minus, smooth_temperature)[0]
        ) / (2 * epsilon)
        smooth_ladder.append(
            {
                "epsilon": epsilon,
                "analytic": smooth_analytic,
                "numerical": numerical,
                "absolute_error": abs(numerical - smooth_analytic),
            }
        )
    return {
        "exact_unique_branch": {"analytic": exact_analytic, "ladder": exact_ladder},
        "smooth_temperature": smooth_temperature,
        "smooth": {"analytic": smooth_analytic, "ladder": smooth_ladder},
        "fresh_verifier_vs_direct": direct_crosscheck,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "official",
        nargs="?",
        type=Path,
        default=ROOT / "external" / "W1.00_2.371339.mat",
    )
    parser.add_argument(
        "candidate",
        nargs="?",
        type=Path,
        default=ROOT / "certificate" / "p4_region_epigraph_stage4.npz",
    )
    parser.add_argument(
        "--output", type=Path, default=ROOT / "certificate" / "W1.00_2.371310_safe_rebuilt.mat"
    )
    parser.add_argument("--margin", type=float, default=1e-10)
    parser.add_argument("--omega-margin", type=float, default=1e-10)
    parser.add_argument("--seed", type=int, default=20260809)
    args = parser.parse_args()

    source = np.load(args.candidate, allow_pickle=True)
    rows = np.asarray(source["optimized_region_weights"], dtype=float)
    official_rows = np.asarray(source["official_region_weights"], dtype=float)
    if rows.shape != official_rows.shape:
        raise ValueError("candidate row arrays have different shapes")
    support = official_rows > 1e-14
    row_checks = {
        "shape": list(rows.shape),
        "minimum_supported": float(rows[support].min()),
        "maximum_unsupported": float(np.max(np.abs(rows[~support]), initial=0.0)),
        "maximum_simplex_residual": float(np.max(np.abs(rows.sum(axis=1) - 1.0))),
        "maximum_row_change": float(np.max(np.abs(rows - official_rows))),
    }
    if row_checks["maximum_unsupported"] > 1e-14 or row_checks["maximum_simplex_residual"] > 2e-13:
        raise RuntimeError(f"invalid candidate rows: {row_checks}")

    model = Model(5, 3, 1)
    model.load(args.official)
    original_params = model.pm.x.copy()
    interior_indices = install_rows(model, rows)
    inferred = model.evaluate()
    candidate_claim = float(np.asarray(source["omega"]))
    direct_difference = float(inferred["tightened_omega"] - candidate_claim)
    if abs(direct_difference) > 5e-11:
        raise RuntimeError(
            f"fresh verifier and candidate objective differ by {direct_difference:.3e}"
        )

    reset_auxiliaries(model, inferred, args.margin, args.omega_margin)
    checked = model.evaluate()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    savemat(args.output, {"params": model.pm.x.reshape(-1, 1)}, do_compression=True)

    # Reloading prevents accidental reliance on cached term state.
    fresh = Model(5, 3, 1)
    fresh.load(args.output)
    reloaded = fresh.evaluate()
    digest = hashlib.sha256(args.output.read_bytes()).hexdigest()

    allowed = np.zeros_like(model.pm.x, dtype=bool)
    for term in model.terms[3]:
        if isinstance(term, TermInfo):
            allowed[term.region_prop_id.sl] = True
    for region in range(6):
        allowed[model.num_retain_comp_id[region][2].sl] = True
        allowed[model.num_retain_comp_id[region][3].sl] = True
        allowed[model.num_retain_glob_id[region].sl] = True
    allowed[model.single_mat_size_id.sl] = True
    allowed[model.omega_id.sl] = True
    unexpected_change = float(
        np.max(np.abs(model.pm.x[~allowed] - original_params[~allowed]), initial=0.0)
    )

    audit = gradient_audit(args.official, rows, args.seed)
    report = {
        "certificate": str(args.output),
        "sha256": digest,
        "parameter_count": int(len(model.pm.x)),
        "interior_row_count": len(interior_indices),
        "candidate_claim": candidate_claim,
        "fresh_tightened_before_aux_reset": float(inferred["tightened_omega"]),
        "candidate_direct_difference": direct_difference,
        "stored_feasible_omega": float(reloaded["omega"]),
        "reloaded_tightened_omega": float(reloaded["tightened_omega"]),
        "stored_record": 2.3713389005434182,
        "beats_stored_record": float(reloaded["omega"]) < 2.3713389005434182,
        "improvement_over_stored_record": 2.3713389005434182 - float(reloaded["omega"]),
        "safety_margin": args.margin,
        "omega_margin": args.omega_margin,
        "max_inequality": float(reloaded["max_ineq"]),
        "max_equality": float(reloaded["max_eq"]),
        "max_bound": float(reloaded["max_bound"]),
        "max_violation": float(reloaded["max_violation"]),
        "row_checks": row_checks,
        "unexpected_nonrow_nonaux_change": unexpected_change,
        "gradient_audit": audit,
        "top_constraints": ranked_constraints(reloaded),
        "checked_vs_reloaded": {
            "omega": float(checked["omega"] - reloaded["omega"]),
            "tightened": float(checked["tightened_omega"] - reloaded["tightened_omega"]),
            "max_violation": float(checked["max_violation"] - reloaded["max_violation"]),
        },
    }
    args.output.with_suffix(".audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
