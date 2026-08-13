#!/usr/bin/env python3
"""Small deterministic CI gate for the published certificate."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

from dual_safe_verifier import evaluate as evaluate_dual
from python_verifier import Model


ROOT = Path(__file__).resolve().parent.parent
CERTIFICATE = ROOT / "certificate" / "W1.00_2.371301_safe_final.mat"
STRUCTURAL_CANDIDATE = (
    ROOT / "certificate" / "joint_global_region_epigraph_final.npz"
)
EXPECTED_SHA256 = "7d68936e6aab7fe4a76073e0036ee06cd8253dcd8d3ab9a0d5689362fdec98f3"
EXPECTED_CANDIDATE_SHA256 = (
    "0bbe4e4946884f480b2aa8b56bb8cc108e94280fa3ca694592ce78a773d8447c"
)
PUBLIC_BOUND = 2.371301


def main() -> None:
    digest = hashlib.sha256(CERTIFICATE.read_bytes()).hexdigest()
    assert digest == EXPECTED_SHA256, (digest, EXPECTED_SHA256)
    candidate_digest = hashlib.sha256(STRUCTURAL_CANDIDATE.read_bytes()).hexdigest()
    assert candidate_digest == EXPECTED_CANDIDATE_SHA256, (
        candidate_digest,
        EXPECTED_CANDIDATE_SHA256,
    )

    model = Model(5, 3, 1.0)
    model.load(CERTIFICATE)
    ordinary = model.evaluate()
    dual = evaluate_dual(CERTIFICATE, PUBLIC_BOUND)

    assert model.pm.num_input == 24_855
    for key in ("omega", "tightened_omega", "max_ineq", "max_eq", "max_bound"):
        assert math.isfinite(float(ordinary[key])), (key, ordinary[key])
    for key in (
        "dual_safe_tightened_omega",
        "dual_safe_log_margin_at_stored_omega",
        "margin_below_public_bound",
    ):
        assert math.isfinite(float(dual[key])), (key, dual[key])
    assert float(ordinary["omega"]) < PUBLIC_BOUND
    assert float(ordinary["tightened_omega"]) < PUBLIC_BOUND
    assert float(ordinary["max_ineq"]) <= -5e-11
    assert float(ordinary["max_eq"]) <= 5e-13
    assert float(ordinary["max_bound"]) <= 0.0
    assert int(dual["entropy_problem_count"]) == 762
    assert float(dual["dual_safe_tightened_omega"]) < PUBLIC_BOUND
    assert float(dual["margin_below_public_bound"]) > 1e-7
    assert float(dual["dual_safe_log_margin_at_stored_omega"]) > 1e-10

    print(
        json.dumps(
            {
                "certificate_sha256": digest,
                "structural_candidate_sha256": candidate_digest,
                "parameter_count": model.pm.num_input,
                "stored_omega": float(ordinary["omega"]),
                "tightened_omega": float(ordinary["tightened_omega"]),
                "dual_safe_omega": float(dual["dual_safe_tightened_omega"]),
                "max_inequality": float(ordinary["max_ineq"]),
                "max_equality": float(ordinary["max_eq"]),
                "max_bound": float(ordinary["max_bound"]),
                "public_bound": PUBLIC_BOUND,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
