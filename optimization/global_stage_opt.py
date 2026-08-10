#!/usr/bin/env python3
"""Fast restricted optimizer for the square-multiplication certificate.

The internal level-3/level-2 policies are frozen at the official certificate.
For fixed policies every contribution below the global stage is linear in the
global shape distribution.  This script caches those coefficients, reducing a
0.5-second 24,855-parameter verification to a ~millisecond objective over one
45-cell distribution (the six regions are related by coordinate permutations).

The max-entropy correction distribution is recomputed by iterative
proportional fitting, rather than treated as an extra optimization variable.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
from scipy.io import savemat
from scipy.optimize import minimize

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "verification"))
from python_verifier import Model, PERMS, TermInfo, entropy, margins, normalized_entropy  # noqa: E402


def max_entropy_same_margins(dist: np.ndarray, shapes: np.ndarray, total: int,
                             tol: float = 2e-14, max_iter: int = 50_000) -> np.ndarray:
    """Maximum-entropy distribution on ``shapes`` with ``dist`` marginals.

    Cyclic iterative proportional fitting converges to the I-projection of the
    uniform distribution, which is precisely the maximum-entropy point.
    """
    targets = margins(dist, shapes, total)
    x = np.ones(len(shapes), dtype=float)
    x /= x.sum()
    for it in range(max_iter):
        old = x.copy()
        for d in range(3):
            current = np.bincount(shapes[:, d], weights=x, minlength=total + 1)
            scale = np.zeros(total + 1)
            good = current > 0
            scale[good] = targets[d][good] / current[good]
            x *= scale[shapes[:, d]]
        if np.max(np.abs(x - old)) < tol:
            break
    else:
        raise RuntimeError("IPF failed to converge")
    x = np.maximum(x, 0)
    x /= x.sum()
    return x


class CachedGlobalObjective:
    def __init__(self, model: Model):
        self.model = model
        self.gs = model.global_stage
        # Populate all fractions and outputs once.
        baseline = model.evaluate()
        self.baseline = baseline
        self.shapes = self.gs.shapes
        self.nshape = len(self.shapes)
        self.lookup = {tuple(s): i for i, s in enumerate(self.shapes)}
        self.perm_index = []
        for perm in PERMS:
            self.perm_index.append(np.asarray([self.lookup[tuple(shape[perm])] for shape in self.shapes]))

        # One row for every global-region / shape top term.
        n_top = 6 * self.nshape
        self.top_shape = np.tile(self.shapes, (6, 1))
        self.top_outer_region = np.repeat(np.arange(6), self.nshape)
        self.top_frac0 = np.concatenate([self.gs.dist[r] * self.gs.region_prop[r] for r in range(6)])
        self.top_csd = [[None] * 3 for _ in range(n_top)]
        self.lv3_nb = np.zeros((n_top, 6, 3))
        self.lv3_pen = np.zeros((n_top, 6))
        self.lv3_py = np.zeros((n_top, 6))
        self.lv3_pz = np.zeros((n_top, 6))
        self.lv2_nb = np.zeros((n_top, 3))
        self.mat = np.zeros((n_top, 3))

        for top_idx, term in enumerate(model.terms[3]):
            frac = self.top_frac0[top_idx]
            if frac <= 0:
                raise ValueError("cache construction expects positive global distribution")
            for d in range(3):
                self.top_csd[top_idx][d] = term.complete_split[d].copy()
            if isinstance(term, TermInfo):
                for r in range(6):
                    self.lv3_nb[top_idx, r] = term.num_block_contribution[r] / frac
                    self.lv3_pen[top_idx, r] = term.hash_penalty_term[r] / frac
                    self.lv3_py[top_idx, r] = term.p_compY[r] / frac
                    self.lv3_pz[top_idx, r] = term.p_compZ[r] / frac
            self.mat[top_idx] += term.mat_size_contribution / frac
        for term in model.terms[2]:
            # For max_level=3 every level-2 identifier is (top term id, region).
            top_idx = int(term.identifier[0]) - 1
            frac = self.top_frac0[top_idx]
            self.lv2_nb[top_idx] += term.num_block_contribution / frac
            self.mat[top_idx] += term.mat_size_contribution / frac
        self.top_csd = [[np.asarray(v) for v in row] for row in self.top_csd]
        self.target = math.log(model.q + 2) * 4

    def symmetric_distributions(self, base: np.ndarray) -> np.ndarray:
        # perm_index[r] maps base entries into permuted shape positions.
        result = np.zeros((6, self.nshape))
        for r in range(6):
            result[r, self.perm_index[r]] = base
        return result

    def global_region_retention(self, r: int, dist: np.ndarray, details=False):
        """Unweighted global retention summary for one orientation.

        The caller supplies the region weight.  In the original program it is
        1/6, so this method returns six times that region's contribution.
        """
        dist = np.asarray(dist, dtype=float)
        dist_max = max_entropy_same_margins(dist, self.shapes, self.gs.sum_col)
        mar = margins(dist, self.shapes, self.gs.sum_col)
        nb = np.asarray([entropy(v) for v in mar])
        penalty = entropy(dist_max) - entropy(dist)
        csd = []
        row0 = r * self.nshape
        for d in range(3):
            csd.append(sum(dist[i] * self.top_csd[row0 + i][d] for i in range(self.nshape)))
        dx, dy, dz = PERMS[r]

        def pcomp(target, is_y):
            weighted = [np.zeros_like(csd[target]) for _ in range(self.gs.sum_col + 1)]
            probs = np.zeros(self.gs.sum_col + 1)
            direct = 0.0
            for i, shape in enumerate(self.shapes):
                if shape[target] == 0:
                    continue
                direct_case = shape[dz] == 0 if is_y else shape.min() == 0
                term_csd = self.top_csd[row0 + i][target]
                if direct_case:
                    direct += dist[i] * entropy(term_csd)
                else:
                    key = int(shape[target])
                    weighted[key] += dist[i] * term_csd
                    probs[key] += dist[i]
            numerator = direct + sum(normalized_entropy(weighted[k], probs[k]) for k in range(1, self.gs.sum_col + 1) if probs[k] > 0)
            denominator = entropy(csd[target]) - entropy(mar[target])
            return numerator - denominator

        py, pz = pcomp(dy, True), pcomp(dz, False)
        corrections = {dx: penalty, dy: py, dz: pz}
        candidates = np.asarray([nb[d] - corrections[d] for d in range(3)])
        value = float(np.min(candidates))
        return (value, candidates) if details else value

    def omega(self, dists: np.ndarray, details=False):
        dists = np.asarray(dists, dtype=float).reshape(6, self.nshape)
        if np.min(dists) < 0 or np.max(np.abs(dists.sum(axis=1) - 1)) > 1e-8:
            return 1e3
        # Global region proportions are fixed at 1/6 in the square verifier.
        top_frac = (dists / 6).ravel()

        # Internal level-3 hashing.
        internal_retain = 0.0
        internal_active = []
        for r in range(6):
            nb = np.tensordot(top_frac, self.lv3_nb[:, r, :], axes=1)
            pen = float(top_frac @ self.lv3_pen[:, r])
            py = float(top_frac @ self.lv3_py[:, r])
            pz = float(top_frac @ self.lv3_pz[:, r])
            dx, dy, dz = PERMS[r]
            candidates = nb.copy()
            candidates[dx] -= pen
            candidates[dy] -= py
            candidates[dz] -= pz
            internal_retain += float(np.min(candidates))
            internal_active.append((int(np.argmin(candidates)), candidates))
        nb2 = top_frac @ self.lv2_nb
        internal_retain += float(np.min(nb2))

        mat = top_frac @ self.mat
        single = float(np.min(mat))

        # Global hashing, including max-entropy penalty and complete-split
        # compatibility losses.
        global_retain = 0.0
        global_active = []
        for r in range(6):
            contribution, candidates = self.global_region_retention(r, dists[r], details=True)
            global_retain += contribution / 6
            global_active.append((int(np.argmin(candidates)), candidates / 6))
        result = (self.target - internal_retain - global_retain) / single
        if details:
            return result, {
                "single": single,
                "mat": mat,
                "internal_retain": internal_retain,
                "global_retain": global_retain,
                "lv2_candidates": nb2,
                "internal_active": internal_active,
                "global_active": global_active,
            }
        return result


def softmax(logits):
    logits = np.asarray(logits, dtype=float)
    z = logits - np.max(logits)
    x = np.exp(z)
    return x / x.sum()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("certificate", type=Path)
    parser.add_argument("--maxiter", type=int, default=80)
    parser.add_argument("--ftol", type=float, default=1e-13)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--method", default="L-BFGS-B", choices=["L-BFGS-B", "Powell", "Nelder-Mead"])
    parser.add_argument("--full", action="store_true", help="optimize six independent 45-cell distributions")
    args = parser.parse_args()

    model = Model(q=5, max_level=3, K=1)
    model.load(args.certificate)
    cache = CachedGlobalObjective(model)
    original = np.asarray(model.global_stage.dist)
    print(f"full-verifier baseline : {cache.baseline['tightened_omega']:.12f}")
    print(f"cached baseline        : {cache.omega(original):.12f}")

    # Average the six inverse-permuted distributions for a diagnostic symmetric
    # start.  The official frozen internal policies are not exactly symmetric,
    # so this is not generally the best restricted point.
    unpermuted = np.zeros_like(original)
    for r in range(6):
        unpermuted[r] = original[r, cache.perm_index[r]]
    base0 = unpermuted.mean(axis=0)
    d0 = cache.symmetric_distributions(base0)
    print(f"symmetrized baseline   : {cache.omega(d0):.12f}")

    eval_count = 0
    start_logits = np.log(original).ravel() if args.full else np.log(base0)
    start_value = cache.omega(original) if args.full else cache.omega(d0)
    best = [start_value, start_logits.copy()]

    def objective(logits):
        nonlocal eval_count
        eval_count += 1
        if args.full:
            dists = np.vstack([softmax(row) for row in np.asarray(logits).reshape(6, cache.nshape)])
        else:
            dists = cache.symmetric_distributions(softmax(logits))
        value = cache.omega(dists)
        if value < best[0]:
            best[:] = [value, logits.copy()]
            print(f"eval {eval_count:6d}: omega={value:.12f}", flush=True)
        return value

    options = {"maxiter": args.maxiter}
    if args.method == "L-BFGS-B":
        options.update(ftol=args.ftol, gtol=1e-9, maxls=30, maxfun=100000)
    elif args.method == "Powell":
        options.update(ftol=args.ftol, xtol=1e-9, maxfev=100000)
    result = minimize(objective, start_logits, method=args.method, options=options)
    if args.full:
        dist_best = np.vstack([softmax(row) for row in np.asarray(best[1]).reshape(6, cache.nshape)])
        base_best = np.asarray([])
    else:
        base_best = softmax(best[1])
        dist_best = cache.symmetric_distributions(base_best)
    omega_best, details = cache.omega(dist_best, details=True)
    print(result)
    print(f"best cached omega      : {omega_best:.12f}")
    print(f"gain                    : {cache.baseline['tightened_omega'] - omega_best:.12e}")
    print(f"details                 : {details}")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        savemat(args.output, {"base_dist": base_best, "dists": dist_best, "omega": omega_best})
        print(f"saved {args.output}")


if __name__ == "__main__":
    main()
