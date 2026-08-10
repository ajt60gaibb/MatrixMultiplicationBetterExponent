#!/usr/bin/env python3
"""Reverse-mode polish of all 126 level-3 region rows in the CW^4 record.

This is deliberately independent of the CW^8 evaluators.  The global laws,
all conditional split laws, and all max-entropy witnesses remain fixed; only
the six hashing-region weights of each interior top constituent vary.
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
sys.path[:0] = [str(ROOT / "verification"), str(HERE)]

from python_verifier import Model, PERMS, TermInfo, entropy, normalized_entropy  # noqa: E402
from global_stage_opt import CachedGlobalObjective  # noqa: E402


def softmin(values: np.ndarray, temperature: float) -> tuple[float, np.ndarray]:
    values = np.asarray(values, dtype=float)
    if temperature <= 0:
        best = float(values.min())
        active = values <= best + 3e-11
        return best, active / active.sum()
    raw = np.exp(-(values - values.min()) / temperature)
    weights = raw / raw.sum()
    return float(values.min() - temperature * np.log(raw.sum())), weights


def masked_softmax(logits: np.ndarray, mask: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(np.where(mask, logits, -np.inf), axis=1, keepdims=True)
    raw = np.where(mask, np.exp(shifted), 0.0)
    return raw / raw.sum(axis=1, keepdims=True)


class P4RegionPolisher:
    def __init__(self, model: Model, lower: CachedGlobalObjective):
        self.model, self.lower = model, lower
        self.shapes = np.asarray(lower.shapes, dtype=int)
        self.n = len(self.shapes)
        self.word = 3**4
        self.word_sum = np.asarray(
            [sum((index // (3**place)) % 3 for place in range(4)) for index in range(self.word)]
        )
        self.alpha = np.asarray(lower.gs.dist, dtype=float)
        self.frac = np.asarray(lower.top_frac0, dtype=float)
        self.interior = []
        self.pi0 = []
        self.basis = np.zeros((6 * self.n, 6, 3, self.word))
        for j, term in enumerate(model.terms[3]):
            if isinstance(term, TermInfo):
                self.interior.append(j)
                self.pi0.append(np.asarray(term.region_prop, dtype=float))
                for h in range(6):
                    for d in range(3):
                        self.basis[j, h, d] = term.complete_split_region[h][d]
        self.interior = np.asarray(self.interior, dtype=int)
        self.pi0 = np.asarray(self.pi0)
        self.mask = self.pi0 > 1e-14

        l2_by_region = np.zeros((6 * self.n, 6, 3))
        mat_by_region = np.zeros_like(l2_by_region)
        for term in model.terms[2]:
            j = int(term.identifier[0]) - 1
            h = int(term.identifier[1])
            l2_by_region[j, h] += term.num_block_contribution / self.frac[j]
            mat_by_region[j, h] += term.mat_size_contribution / self.frac[j]
        count = len(self.interior)
        self.raw3 = np.zeros((count, 6, 3))
        self.raw2 = np.zeros_like(self.raw3)
        self.rawmat = np.zeros_like(self.raw3)
        for local, j in enumerate(self.interior):
            for h in range(6):
                if self.pi0[local, h] <= 1e-14:
                    continue
                dx, dy, dz = map(int, PERMS[h])
                for d in range(3):
                    correction = (
                        lower.lv3_pen[j, h] if d == dx else
                        lower.lv3_py[j, h] if d == dy else lower.lv3_pz[j, h]
                    )
                    self.raw3[local, h, d] = (
                        lower.lv3_nb[j, h, d] - correction
                    ) / self.pi0[local, h]
                self.raw2[local, h] = l2_by_region[j, h] / self.pi0[local, h]
                self.rawmat[local, h] = mat_by_region[j, h] / self.pi0[local, h]

        interior_mask = np.zeros(6 * self.n, dtype=bool)
        interior_mask[self.interior] = True
        self.boundary = ~interior_mask
        self._context: dict[str, np.ndarray] = {}

    def forward(self, pi: np.ndarray, details: bool = False):
        pi = np.asarray(pi, dtype=float).reshape(self.pi0.shape)
        csd = np.asarray(self.lower.top_csd, dtype=float).copy()
        for local, j in enumerate(self.interior):
            csd[j] = np.einsum("h,hdw->dw", pi[local], self.basis[j])

        cf = self.frac[self.interior]
        cand3 = np.einsum("j,jh,jhd->hd", cf, pi, self.raw3)
        cand2 = self.frac[self.boundary] @ self.lower.lv2_nb[self.boundary]
        mat = self.frac[self.boundary] @ self.lower.mat[self.boundary]
        cand2 += np.einsum("j,jh,jhd->d", cf, pi, self.raw2)
        mat += np.einsum("j,jh,jhd->d", cf, pi, self.rawmat)

        glob = np.zeros((6, 3))
        for r in range(6):
            dist = self.alpha[r]
            row = csd[r * self.n : (r + 1) * self.n]
            # The anchor branch is independent of pi and already cached exactly.
            _, candidates = self.lower.global_region_retention(r, dist, details=True)
            dx, dy, dz = map(int, PERMS[r])
            glob[r, dx] = candidates[dx] / 6.0
            for target, is_y in ((dy, True), (dz, False)):
                mix = dist @ row[:, target]
                direct = (self.shapes[:, target] != 0) & (
                    (self.shapes[:, dz] == 0) if is_y else (np.min(self.shapes, axis=1) == 0)
                )
                nondirect = (self.shapes[:, target] != 0) & ~direct
                numerator = sum(
                    dist[s] * entropy(row[s, target]) for s in np.flatnonzero(direct)
                )
                for k in range(1, 9):
                    selected = nondirect & (self.shapes[:, target] == k)
                    if not np.any(selected):
                        continue
                    mass = float(dist[selected].sum())
                    numerator += normalized_entropy(dist[selected] @ row[selected, target], mass)
                glob[r, target] = (entropy(mix) - numerator) / 6.0

        retention = float(sum(np.min(row) for row in cand3) + np.min(cand2))
        retention += float(sum(np.min(row) for row in glob))
        single = float(np.min(mat))
        omega = (4.0 * math.log(7.0) - retention) / single
        self._context = {"csd": csd, "cand3": cand3, "cand2": cand2, "glob": glob, "mat": mat}
        detail = {
            "omega": omega, "retention": retention, "single": single,
            "level3": cand3, "level2": cand2, "global": glob, "mat": mat,
            "max_csd_residual": float(np.max(np.abs(csd.sum(axis=2) - 1.0))),
        }
        return (omega, detail) if details else omega

    def value_gradient(self, pi: np.ndarray, temperature: float) -> tuple[float, np.ndarray]:
        self.forward(pi)
        ctx = self._context
        retention = 0.0
        w3 = []
        for row in ctx["cand3"]:
            value, weights = softmin(row, temperature); retention += value; w3.append(weights)
        value2, w2 = softmin(ctx["cand2"], temperature); retention += value2
        wg = []
        for row in ctx["glob"]:
            value, weights = softmin(row, temperature); retention += value; wg.append(weights)
        single, wm = softmin(ctx["mat"], temperature)
        w3, wg = np.asarray(w3), np.asarray(wg)
        omega = (4.0 * math.log(7.0) - retention) / single

        csd = ctx["csd"]
        adj = np.zeros_like(csd)
        for r in range(6):
            dist = self.alpha[r]
            row = csd[r * self.n : (r + 1) * self.n]
            _, dy, dz = map(int, PERMS[r])
            for target, is_y in ((dy, True), (dz, False)):
                if wg[r, target] == 0:
                    continue
                C = row[:, target]
                mix = dist @ C
                local_adj = np.tile(-np.log(np.maximum(mix, 1e-300)) - 1.0, (self.n, 1))
                direct = (self.shapes[:, target] != 0) & (
                    (self.shapes[:, dz] == 0) if is_y else (np.min(self.shapes, axis=1) == 0)
                )
                nondirect = (self.shapes[:, target] != 0) & ~direct
                if np.any(nondirect):
                    weighted = dist[nondirect] @ C[nondirect]
                    probs = np.bincount(
                        self.shapes[nondirect, target], weights=dist[nondirect], minlength=9
                    )
                    positive = weighted > 0
                    correction = np.zeros(self.word)
                    correction[positive] = (
                        np.log(probs[self.word_sum[positive]]) - np.log(weighted[positive])
                    )
                    local_adj[nondirect] -= correction
                for s in np.flatnonzero(direct):
                    positive = C[s] > 0
                    local_adj[s, positive] -= -np.log(C[s, positive]) - 1.0
                adj[r * self.n : (r + 1) * self.n, target] = (
                    wg[r, target] * dist[:, None] * local_adj / 6.0
                )

        cf = self.frac[self.interior]
        gret = np.zeros_like(pi)
        gsize = np.zeros_like(pi)
        for h in range(6):
            gret[:, h] += cf * (self.raw3[:, h] @ w3[h])
        gret += cf[:, None] * np.einsum("jhd,d->jh", self.raw2, w2)
        gsize += cf[:, None] * np.einsum("jhd,d->jh", self.rawmat, wm)
        for local, j in enumerate(self.interior):
            gret[local] += np.einsum("hdw,dw->h", self.basis[j], adj[j])
        gradient = -(gret + omega * gsize) / single
        return omega, gradient

    def install_and_verify(self, pi: np.ndarray) -> dict[str, float]:
        for local, j in enumerate(self.interior):
            group = self.model.terms[3][j].region_prop_id
            self.model.pm.x[group.sl] = pi[local]
        checked = self.model.evaluate()
        return {
            "verifier_tightened_omega": float(checked["tightened_omega"]),
            "verifier_max_equality": float(checked["max_eq"]),
            "verifier_max_inequality": float(checked["max_ineq"]),
            "verifier_max_violation": float(checked["max_violation"]),
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("certificate", type=Path)
    parser.add_argument("--start", type=Path)
    parser.add_argument("--temps", default="1e-7,3e-8,1e-8,3e-9,0")
    parser.add_argument("--maxiter", type=int, default=20)
    parser.add_argument("--output", type=Path, default=HERE / "p4_lower_region_candidate.npz")
    args = parser.parse_args()
    model = Model(5, 3, 1); model.load(args.certificate)
    lower = CachedGlobalObjective(model)
    program = P4RegionPolisher(model, lower)
    pi = program.pi0.copy()
    if args.start:
        start_file = np.load(args.start, allow_pickle=True)
        for key in ("optimized_region_weights", "lower_region_weights"):
            if key in start_file.files:
                pi = np.asarray(start_file[key], dtype=float)
                break
        else:
            raise KeyError("start file needs optimized_region_weights or lower_region_weights")
    logits = np.log(np.maximum(pi, 1e-300))
    start, start_detail = program.forward(pi, details=True)
    best = (start, logits.copy())
    history = []
    print(f"p4 start exact={start:.15f}", flush=True)
    for stage, temperature in enumerate(float(x) for x in args.temps.split(",")):
        calls = 0
        def objective(flat):
            nonlocal calls
            calls += 1
            rows = masked_softmax(flat.reshape(pi.shape), program.mask)
            value, gp = program.value_gradient(rows, temperature)
            gz = rows * (gp - np.sum(rows * gp, axis=1, keepdims=True))
            print(f"T={temperature:.1e} call={calls:03d} value={value:.15f}", flush=True)
            return value, gz.ravel()
        result = minimize(
            objective, logits.ravel(), jac=True, method="L-BFGS-B",
            options={"maxiter": args.maxiter, "ftol": 2e-15, "gtol": 2e-10, "maxls": 30},
        )
        logits = result.x.reshape(pi.shape)
        exact = program.forward(masked_softmax(logits, program.mask))
        if exact < best[0]: best = (exact, logits.copy())
        history.append((temperature, float(result.fun), exact, result.nit, result.nfev, str(result.message)))
        print(f"stage={history[-1]}", flush=True)
    pi = masked_softmax(best[1], program.mask)
    omega, detail = program.forward(pi, details=True)
    verification = program.install_and_verify(pi)
    record = 2.3713389005434182
    report = {
        "omega": omega, "start": start, "gain": start - omega,
        "stored_record": record, "beats_stored_record": omega < record,
        "history": history, "direct_detail": {
            k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in detail.items()
        }, **verification,
    }
    np.savez_compressed(
        args.output, optimized_region_weights=pi, official_region_weights=program.pi0,
        omega=np.asarray(omega), history=np.asarray(history, dtype=object), report=np.asarray(report, dtype=object),
    )
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
