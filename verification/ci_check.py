#!/usr/bin/env python3
"""Small deterministic CI gate for the published certificate."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from dual_safe_verifier import evaluate as evaluate_dual
from python_verifier import Model


ROOT = Path(__file__).resolve().parent.parent
CERTIFICATE = ROOT / "certificate" / "W1.00_2.371310_safe_final.mat"
EXPECTED_SHA256 = "65d78064ef3686f850558b3881446403afe9d4910d2d2e046c3dab92c390ff83"


def main() -> None:
    digest = hashlib.sha256(CERTIFICATE.read_bytes()).hexdigest()
    assert digest == EXPECTED_SHA256, (digest, EXPECTED_SHA256)

    model = Model(5, 3, 1.0)
    model.load(CERTIFICATE)
    ordinary = model.evaluate()
    dual = evaluate_dual(CERTIFICATE)

    assert model.pm.num_input == 24_855
    assert float(ordinary["omega"]) < 2.371310
    assert float(ordinary["max_ineq"]) < 0.0
    assert float(ordinary["max_bound"]) <= 0.0
    assert int(dual["entropy_problem_count"]) == 762
    assert float(dual["dual_safe_tightened_omega"]) < 2.371310
    assert float(dual["dual_safe_log_margin_at_stored_omega"]) > 0.0

    print(
        json.dumps(
            {
                "certificate_sha256": digest,
                "parameter_count": model.pm.num_input,
                "stored_omega": float(ordinary["omega"]),
                "tightened_omega": float(ordinary["tightened_omega"]),
                "dual_safe_omega": float(dual["dual_safe_tightened_omega"]),
                "max_inequality": float(ordinary["max_ineq"]),
                "max_bound": float(ordinary["max_bound"]),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

