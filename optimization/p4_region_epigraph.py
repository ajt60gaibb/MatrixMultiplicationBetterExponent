#!/usr/bin/env python3
"""Epigraph/active-set refinement of the CW^4 lower region rows.

Unlike smooth-min continuation, this program keeps explicit variables for all
six L3 retentions, symmetric L2 retention, six global retentions, and the
single-matrix logarithm.  Forty-two candidate inequalities enforce the exact
nonsmooth minima while SLSQP minimizes the resulting exponent ratio.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
VERIFY = ROOT / "verification"
sys.path[:0] = [str(VERIFY), str(HERE)]

from python_verifier import Model, PERMS, entropy  # noqa: E402
from global_stage_opt import CachedGlobalObjective  # noqa: E402
from p4_lower_region_polish import P4RegionPolisher, masked_softmax  # noqa: E402
from certify_p4_region_candidate import install_rows  # noqa: E402


class EpigraphProgram:
    def __init__(self, polisher: P4RegionPolisher, initial_rows: np.ndarray):
        self.p = polisher
        self.initial_rows = np.asarray(initial_rows, dtype=float)
        self.logit_shape = self.initial_rows.shape
        self.nlogit = self.initial_rows.size
        self.naux = 14
        self.target = 4.0 * math.log(7.0)
        self._last_x: np.ndarray | None = None
        self._last: dict | None = None

    def initial(self, slack: float = 1e-10) -> np.ndarray:
        _, detail = self.p.forward(self.initial_rows, details=True)
        aux = np.r_[
            np.min(detail["level3"], axis=1) - slack,
            float(np.min(detail["level2"])) - slack,
            np.min(detail["global"], axis=1) - slack,
            float(np.min(detail["mat"])) - slack,
        ]
        return np.r_[np.log(np.maximum(self.initial_rows, 1e-300)).ravel(), aux]

    def _global_gradients(self, pi: np.ndarray, csd: np.ndarray) -> np.ndarray:
        """Derivative of every global candidate with respect to every row."""
        count = len(self.p.interior)
        output = np.zeros((6, 3, count, 6))
        for r in range(6):
            dist = self.p.alpha[r]
            row = csd[r * self.p.n : (r + 1) * self.p.n]
            _, dy, dz = map(int, PERMS[r])
            for target, is_y in ((dy, True), (dz, False)):
                C = row[:, target]
                mixture = dist @ C
                local_adjoint = np.tile(
                    -np.log(np.maximum(mixture, 1e-300)) - 1.0,
                    (self.p.n, 1),
                )
                direct = (self.p.shapes[:, target] != 0) & (
                    (self.p.shapes[:, dz] == 0)
                    if is_y
                    else (np.min(self.p.shapes, axis=1) == 0)
                )
                nondirect = (self.p.shapes[:, target] != 0) & ~direct
                if np.any(nondirect):
                    weighted = dist[nondirect] @ C[nondirect]
                    probabilities = np.bincount(
                        self.p.shapes[nondirect, target],
                        weights=dist[nondirect],
                        minlength=9,
                    )
                    positive = weighted > 0
                    correction = np.zeros(self.p.word)
                    correction[positive] = (
                        np.log(probabilities[self.p.word_sum[positive]])
                        - np.log(weighted[positive])
                    )
                    local_adjoint[nondirect] -= correction
                for shape_index in np.flatnonzero(direct):
                    positive = C[shape_index] > 0
                    local_adjoint[shape_index, positive] -= (
                        -np.log(C[shape_index, positive]) - 1.0
                    )
                adjoint = dist[:, None] * local_adjoint / 6.0
                for local, child in enumerate(self.p.interior):
                    if child // self.p.n != r:
                        continue
                    output[r, target, local] = np.einsum(
                        "hdw,dw->h",
                        self.p.basis[child, :, target : target + 1],
                        adjoint[child % self.p.n : child % self.p.n + 1],
                    )
        return output

    @staticmethod
    def _softmax_pullback(pi: np.ndarray, gradients: np.ndarray) -> np.ndarray:
        return pi[None, :, :] * (
            gradients - np.sum(gradients * pi[None, :, :], axis=2, keepdims=True)
        )

    def compute(self, x: np.ndarray) -> dict:
        x = np.asarray(x, dtype=float)
        if self._last_x is not None and np.array_equal(x, self._last_x):
            assert self._last is not None
            return self._last
        logits = x[: self.nlogit].reshape(self.logit_shape)
        pi = masked_softmax(logits, self.p.mask)
        _, detail = self.p.forward(pi, details=True)
        context = self.p._context
        aux = x[self.nlogit :]
        e3 = aux[:6]
        e2 = aux[6]
        eg = aux[7:13]
        single = aux[13]
        retained = float(e3.sum() + e2 + eg.sum())
        objective = (self.target - retained) / single

        candidates = np.r_[
            (detail["level3"] - e3[:, None]).ravel(),
            detail["level2"] - e2,
            (detail["global"] - eg[:, None]).ravel(),
            detail["mat"] - single,
        ]

        nconstraint = len(candidates)
        row_gradients = np.zeros((nconstraint, *self.logit_shape))
        cf = self.p.frac[self.p.interior]
        cursor = 0
        for region in range(6):
            for dimension in range(3):
                row_gradients[cursor, :, region] = (
                    cf * self.p.raw3[:, region, dimension]
                )
                cursor += 1
        for dimension in range(3):
            row_gradients[cursor] = (
                cf[:, None] * self.p.raw2[:, :, dimension]
            )
            cursor += 1
        global_gradients = self._global_gradients(pi, context["csd"])
        for region in range(6):
            for dimension in range(3):
                row_gradients[cursor] = global_gradients[region, dimension]
                cursor += 1
        for dimension in range(3):
            row_gradients[cursor] = (
                cf[:, None] * self.p.rawmat[:, :, dimension]
            )
            cursor += 1
        assert cursor == nconstraint
        logit_jacobian = self._softmax_pullback(pi, row_gradients).reshape(
            nconstraint, self.nlogit
        )
        jacobian = np.zeros((nconstraint, self.nlogit + self.naux))
        jacobian[:, : self.nlogit] = logit_jacobian
        cursor = 0
        for region in range(6):
            jacobian[cursor : cursor + 3, self.nlogit + region] = -1.0
            cursor += 3
        jacobian[cursor : cursor + 3, self.nlogit + 6] = -1.0
        cursor += 3
        for region in range(6):
            jacobian[cursor : cursor + 3, self.nlogit + 7 + region] = -1.0
            cursor += 3
        jacobian[cursor : cursor + 3, self.nlogit + 13] = -1.0

        objective_gradient = np.zeros(self.nlogit + self.naux)
        objective_gradient[self.nlogit : self.nlogit + 13] = -1.0 / single
        objective_gradient[self.nlogit + 13] = -objective / single
        result = {
            "pi": pi,
            "detail": detail,
            "objective": objective,
            "objective_gradient": objective_gradient,
            "constraints": candidates,
            "constraint_jacobian": jacobian,
        }
        self._last_x = x.copy()
        self._last = result
        return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "certificate", type=Path, default=ROOT / "certificate" / "W1.00_2.371310_safe_final.mat", nargs="?"
    )
    parser.add_argument(
        "rows", type=Path, default=ROOT / "certificate" / "p4_region_epigraph_stage4.npz", nargs="?"
    )
    parser.add_argument("--maxiter", type=int, default=200)
    parser.add_argument("--ftol", type=float, default=1e-13)
    parser.add_argument("--constraint-scale", type=float, default=1.0)
    parser.add_argument("--patience", type=int, default=200)
    parser.add_argument("--check-gradient", action="store_true")
    parser.add_argument("--seed", type=int, default=20260809)
    parser.add_argument("--output", type=Path, default=HERE / "p4_region_epigraph_candidate.npz")
    args = parser.parse_args()

    model = Model(5, 3, 1)
    model.load(args.certificate)
    lower = CachedGlobalObjective(model)
    polisher = P4RegionPolisher(model, lower)
    saved = np.load(args.rows, allow_pickle=True)
    rows = np.asarray(saved["optimized_region_weights"], dtype=float)
    program = EpigraphProgram(polisher, rows)
    x = program.initial()
    start = program.compute(x)
    print(
        f"start objective={start['objective']:.15f} "
        f"min_constraint={start['constraints'].min():.3e} variables={len(x)}",
        flush=True,
    )

    if args.check_gradient:
        rng = np.random.default_rng(args.seed)
        direction = rng.standard_normal(len(x))
        direction[program.nlogit :] *= 1e-3
        direction /= np.linalg.norm(direction)
        epsilon = 1e-5
        plus = program.compute(x + epsilon * direction)
        minus = program.compute(x - epsilon * direction)
        objective_fd = (plus["objective"] - minus["objective"]) / (2 * epsilon)
        objective_ad = float(start["objective_gradient"] @ direction)
        constraint_fd = (plus["constraints"] - minus["constraints"]) / (2 * epsilon)
        constraint_ad = start["constraint_jacobian"] @ direction
        print(
            "gradient_audit="
            + repr(
                {
                    "objective_analytic": objective_ad,
                    "objective_numerical": objective_fd,
                    "objective_error": abs(objective_ad - objective_fd),
                    "constraint_max_error": float(np.max(np.abs(constraint_ad - constraint_fd))),
                }
            ),
            flush=True,
        )

    calls = 0
    start_direct = float(polisher.forward(rows))
    best_direct = [start_direct, rows.copy(), x.copy(), 0]

    class StopSearch(RuntimeError):
        pass

    def direct_from_detail(detail: dict) -> float:
        retention = float(np.min(detail["level3"], axis=1).sum())
        retention += float(np.min(detail["level2"]))
        retention += float(np.min(detail["global"], axis=1).sum())
        return (4.0 * math.log(7.0) - retention) / float(np.min(detail["mat"]))

    def objective(value: np.ndarray):
        nonlocal calls
        calls += 1
        result = program.compute(value)
        exact = direct_from_detail(result["detail"])
        if exact < best_direct[0] - 2e-14:
            best_direct[:] = [exact, result["pi"].copy(), value.copy(), calls]
            np.savez_compressed(
                args.output.with_name(args.output.stem + ".best_direct_checkpoint.npz"),
                optimized_region_weights=best_direct[1],
                omega=np.asarray(best_direct[0]),
                call=np.asarray(calls),
            )
        if calls % 10 == 0:
            print(
                f"call={calls:04d} epigraph={result['objective']:.15f} "
                f"direct={exact:.15f} best={best_direct[0]:.15f} "
                f"minc={result['constraints'].min():+.3e}",
                flush=True,
            )
        if calls - int(best_direct[3]) >= args.patience:
            raise StopSearch(f"no direct improvement for {args.patience} evaluations")
        return result["objective"], result["objective_gradient"]

    constraints = {
        "type": "ineq",
        "fun": lambda value: args.constraint_scale * program.compute(value)["constraints"],
        "jac": lambda value: args.constraint_scale * program.compute(value)["constraint_jacobian"],
    }
    bounds = [(None, None)] * program.nlogit + [(0.0, None)] * program.naux
    stopped = None
    try:
        result = minimize(
            objective,
            x,
            jac=True,
            constraints=constraints,
            bounds=bounds,
            method="SLSQP",
            options={"maxiter": args.maxiter, "ftol": args.ftol, "disp": True},
        )
        final_x = result.x
    except StopSearch as error:
        stopped = str(error)
        result = None
        final_x = np.asarray(best_direct[2])
    final = program.compute(final_x)
    # The epigraph auxiliaries can be slightly infeasible while the row point
    # is entirely valid.  Always emit the best independently evaluated rows.
    final_pi = np.asarray(best_direct[1])
    direct = float(best_direct[0])
    fresh = Model(5, 3, 1)
    fresh.load(args.certificate)
    install_rows(fresh, final_pi)
    verified = fresh.evaluate()
    report = {
        "direct_omega": direct,
        "epigraph_omega": final["objective"],
        "fresh_verifier_tightened": float(verified["tightened_omega"]),
        "start_direct": start_direct,
        "gain": start_direct - direct,
        "minimum_constraint": float(final["constraints"].min()),
        "success": bool(result.success) if result is not None else False,
        "message": str(result.message) if result is not None else stopped,
        "iterations": int(result.nit) if result is not None else None,
        "evaluations": int(result.nfev) if result is not None else calls,
        "best_direct_call": int(best_direct[3]),
        "constraint_scale": args.constraint_scale,
        "stored_record": 2.3713389005434182,
        "beats_record": direct < 2.3713389005434182,
    }
    np.savez_compressed(
        args.output,
        optimized_region_weights=final_pi,
        official_region_weights=polisher.pi0,
        omega=np.asarray(direct),
        epigraph_variables=final_x[program.nlogit :],
        constraints=final["constraints"],
        report=np.asarray(report, dtype=object),
    )
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
