#!/usr/bin/env python3
"""Conservative max-entropy-dual audit for the omega < 2.371310 certificate.

The released verifier represents each maximum-entropy distribution by a
primal distribution and KKT multipliers.  This audit does not assume that the
floating-point KKT equalities are exact.  Instead, it evaluates the entropy
dual at the stored multipliers.  Weak duality makes every resulting value an
upper bound for the relevant maximum entropy, which is the conservative
direction in every hash-penalty term.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
MORE = HERE.parent / "more_asymmetry"
if not (MORE / "python_verifier.py").exists():
    # The publication bundle places both verifiers in one ancillary folder.
    MORE = HERE
sys.path.insert(0, str(MORE))

from python_verifier import Model, PERMS, TermInfo, entropy, margins  # noqa: E402


def term_dual_entropy(term: TermInfo, region: int) -> float:
    exponential_sum = 0.0
    for split in term.splits:
        left = split[:3]
        exponent = term.lam_sum[region] - 1.0
        for dimension in range(3):
            exponent += term.lam_margin[region][dimension][
                left[dimension] - term.lam_low[dimension]
            ]
        exponential_sum += math.exp(exponent)

    marginal = margins(term.split_dist[region], term.splits[:, :3], term.sum_half)
    value = exponential_sum - term.lam_sum[region]
    for dimension in range(3):
        support = np.arange(term.lam_low[dimension], term.lam_high[dimension] + 1)
        value -= float(
            np.dot(
                term.lam_margin[region][dimension],
                marginal[dimension][support],
            )
        )
    return value


def global_dual_entropy(model: Model, region: int) -> float:
    stage = model.global_stage
    exponential_sum = 0.0
    for shape in stage.shapes:
        exponent = stage.lam_sum[region] - 1.0
        for dimension in range(3):
            exponent += stage.lam_margin[region][dimension][shape[dimension]]
        exponential_sum += math.exp(exponent)

    marginal = margins(stage.dist[region], stage.shapes, stage.sum_col)
    value = exponential_sum - stage.lam_sum[region]
    for dimension in range(3):
        value -= float(np.dot(stage.lam_margin[region][dimension], marginal[dimension]))
    return value


def evaluate(certificate: Path) -> dict:
    model = Model(5, 3, 1.0)
    model.load(certificate)
    ordinary = model.evaluate()

    primal_dual_differences: list[float] = []
    retained = 0.0

    for level in range(3, model.max_level + 1):
        for region in range(6):
            dim_x, dim_y, dim_z = PERMS[region]
            num_block = sum(
                (term.num_block_contribution[region] for term in model.terms[level]),
                np.zeros(3),
            )
            penalty = 0.0
            for term in model.terms[level]:
                if not isinstance(term, TermInfo):
                    continue
                dual_entropy = term_dual_entropy(term, region)
                primal_dual_differences.append(
                    dual_entropy - entropy(term.split_dist_max[region])
                )
                penalty += (
                    dual_entropy - entropy(term.split_dist[region])
                ) * term.term_frac * term.region_prop[region]
            p_y = sum(term.p_compY[region] for term in model.terms[level])
            p_z = sum(term.p_compZ[region] for term in model.terms[level])
            candidates = []
            for dimension in range(3):
                correction = penalty if dimension == dim_x else p_y if dimension == dim_y else p_z
                candidates.append(num_block[dimension] - correction)
            retained += min(candidates)

    level_two = sum(
        (term.num_block_contribution for term in model.terms[2]),
        np.zeros(3),
    )
    retained += float(np.min(level_two))

    stage = model.global_stage
    for region in range(6):
        dim_x, dim_y, dim_z = PERMS[region]
        dual_entropy = global_dual_entropy(model, region)
        primal_dual_differences.append(dual_entropy - entropy(stage.dist_max[region]))
        penalty = (
            dual_entropy - entropy(stage.dist[region])
        ) * stage.region_prop[region]
        corrections = {
            dim_x: penalty,
            dim_y: stage.p_compY[region],
            dim_z: stage.p_compZ[region],
        }
        retained += min(
            stage.num_block[region][dimension] - corrections[dimension]
            for dimension in range(3)
        )

    single = float(np.min(stage.mat_size))
    target = 4.0 * math.log(7.0)
    dual_safe_omega = (target - retained) / single
    stored_omega = float(ordinary["omega"])
    return {
        "certificate": str(certificate),
        "entropy_problem_count": len(primal_dual_differences),
        "minimum_dual_minus_stored_primal_entropy": min(primal_dual_differences),
        "maximum_dual_minus_stored_primal_entropy": max(primal_dual_differences),
        "ordinary_tightened_omega": float(ordinary["tightened_omega"]),
        "dual_safe_tightened_omega": dual_safe_omega,
        "stored_safe_omega": stored_omega,
        "margin_below_2.371310": 2.371310 - dual_safe_omega,
        "dual_safe_log_margin_at_stored_omega": retained + single * stored_omega - target,
        "dual_formula": "sum_s exp((A^T lambda)_s-1) - <lambda,mu>",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("certificate", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = evaluate(args.certificate)
    encoded = json.dumps(result, indent=2, sort_keys=True)
    print(encoded)
    if args.output:
        args.output.write_text(encoded + "\n")


if __name__ == "__main__":
    main()
