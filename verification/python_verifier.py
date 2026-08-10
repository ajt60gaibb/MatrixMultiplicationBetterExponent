#!/usr/bin/env python3
"""Independent NumPy/SciPy verifier for More Asymmetry parameters.

This is a value-only port of the authors' MATLAB verifier.  It deliberately
does not use their GVar automatic differentiation class or SNOPT, so it gives
an independent check of both the parameter-vector layout and every numerical
inequality used by ``VerifyOmega.m``.

The implementation follows the registration/evaluation order of:

  * src/evaluation/{Workspace,GlobalStage,TermInfo,TermInfoLv2,TermInfoZero}.m
  * src/utils and src/complete_split

Only the square-multiplication verification mode is implemented.  The code is
written for arbitrary ``max_level``; the supplied certificates use level 3.
"""

from __future__ import annotations

import argparse
import dataclasses
import math
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from scipy.io import loadmat


PERMS = np.asarray(
    [[0, 1, 2], [0, 2, 1], [1, 0, 2], [1, 2, 0], [2, 0, 1], [2, 1, 0]],
    dtype=int,
)


def entropy(x: np.ndarray | Sequence[float]) -> float:
    x = np.asarray(x, dtype=float)
    positive = x > 0
    return float(-np.dot(x[positive], np.log(x[positive])))


def normalized_entropy(x: np.ndarray, p: float) -> float:
    x = np.asarray(x, dtype=float)
    if p <= 0:
        return 0.0
    positive = x > 0
    return float(-np.dot(x[positive], np.log(x[positive] / p)))


def prepare_shapes(level: int) -> np.ndarray:
    total = 2**level
    return np.asarray(
        [(i, j, total - i - j) for i in range(total + 1) for j in range(total - i + 1)],
        dtype=int,
    )


def prepare_splits(shape: Sequence[int]) -> np.ndarray:
    shape = np.asarray(shape, dtype=int)
    half = int(shape.sum() // 2)
    result = []
    for i in range(min(half, int(shape[0])) + 1):
        for j in range(min(half - i, int(shape[1])) + 1):
            k = half - i - j
            if k <= shape[2]:
                left = np.asarray([i, j, k], dtype=int)
                result.append(np.r_[left, shape - left])
    return np.asarray(result, dtype=int)


def margins(dist: np.ndarray, support: np.ndarray, total: int) -> tuple[np.ndarray, ...]:
    out = []
    for d in range(3):
        out.append(np.bincount(support[:, d], weights=dist, minlength=total + 1))
    return tuple(out)


def decode_csd(index: int, power: int) -> np.ndarray:
    result = np.zeros(power, dtype=int)
    for pos in range(power - 1, -1, -1):
        result[pos] = index % 3
        index //= 3
    return result


def encode_csd(values: Sequence[int]) -> int:
    result = 0
    for value in values:
        result = result * 3 + int(value)
    return result


def rot3(x: Sequence, count: int):
    count %= 3
    if count == 0:
        return x
    if count == 1:
        return [x[1], x[2], x[0]]
    return [x[2], x[0], x[1]]


@dataclasses.dataclass(frozen=True)
class Group:
    start: int
    size: int
    lower: np.ndarray
    upper: np.ndarray
    label: str

    @property
    def sl(self) -> slice:
        return slice(self.start, self.start + self.size)


class ParameterManager:
    def __init__(self):
        self.groups: list[Group] = []
        self.num_input = 0
        self.linear_equalities: list[tuple[list[tuple[Group, np.ndarray]], np.ndarray, str]] = []
        self.x: np.ndarray | None = None

    def register(self, size: int, lower=-np.inf, upper=np.inf, label="") -> Group:
        lo = np.broadcast_to(np.asarray(lower, dtype=float), (size,)).copy()
        hi = np.broadcast_to(np.asarray(upper, dtype=float), (size,)).copy()
        group = Group(self.num_input, size, lo, hi, label)
        self.groups.append(group)
        self.num_input += size
        return group

    def add_eq(self, entries: list[tuple[Group, np.ndarray]], rhs, label=""):
        self.linear_equalities.append((entries, np.atleast_1d(np.asarray(rhs, dtype=float)), label))

    def get(self, group: Group) -> np.ndarray:
        assert self.x is not None
        return self.x[group.sl]

    def linear_residuals(self) -> list[tuple[str, np.ndarray]]:
        residuals = []
        for entries, rhs, label in self.linear_equalities:
            value = np.zeros_like(rhs)
            for group, coeff in entries:
                coeff = np.asarray(coeff, dtype=float)
                # MATLAB stores a (group-size) x (number constraints) matrix.
                if coeff.ndim == 0:
                    coeff = coeff.reshape(1, 1)
                elif coeff.ndim == 1:
                    coeff = coeff.reshape(-1, 1)
                value += self.get(group) @ coeff
            residuals.append((label, value - rhs))
        return residuals

    def bound_violation(self) -> float:
        assert self.x is not None
        best = 0.0
        for group in self.groups:
            value = self.get(group)
            best = max(best, float(np.max(group.lower - value)), float(np.max(value - group.upper)))
        return best


class BaseTerm:
    def __init__(self, model: "Model", level: int, term_id: int, shape, identifier):
        self.model = model
        self.pm = model.pm
        self.level = level
        self.power = 2 ** (level - 1)
        self.sum_col = 2**level
        self.sum_half = self.sum_col // 2
        self.term_id = term_id
        self.shape = np.asarray(shape, dtype=int)
        self.identifier = tuple(identifier)
        self.term_frac = 0.0

    def evaluate_init(self):
        self.term_frac = 0.0

    def evaluate_pre(self):
        pass


class TermInfoLv2(BaseTerm):
    def __init__(self, model, level, term_id, shape, identifier):
        super().__init__(model, level, term_id, shape, identifier)
        if int(self.shape.max()) == 2 and int(self.shape.min()) == 0:
            self.shape_type, standard = "022", np.asarray([0, 2, 2])
        elif int(self.shape.max()) == 2:
            self.shape_type, standard = "112", np.asarray([1, 1, 2])
        elif int(self.shape.max()) == 3:
            if tuple(self.shape) in ((0, 1, 3), (1, 3, 0), (3, 0, 1)):
                self.shape_type, standard = "013", np.asarray([0, 1, 3])
            else:
                self.shape_type, standard = "031", np.asarray([0, 3, 1])
        else:
            self.shape_type, standard = "004", np.asarray([0, 0, 4])
        work = self.shape.copy()
        self.rotate_num = 0
        while not np.array_equal(work, standard):
            work = np.asarray([work[2], work[0], work[1]])
            self.rotate_num += 1
        self.split_0_id = None
        if self.shape_type in ("112", "022"):
            self.split_0_id = self.pm.register(1, 0, 0.5, f"lv2[{term_id}].split0")

    def evaluate_init(self):
        super().evaluate_init()
        self.split_0 = float(self.pm.get(self.split_0_id)[0]) if self.split_0_id else None

    def evaluate_post(self):
        q = self.model.q
        csd = [np.zeros(9), np.zeros(9), np.zeros(9)]
        if self.shape_type == "022":
            s = self.split_0
            num_block = np.zeros(3)
            inner = entropy([s, s, 1 - 2 * s]) + 2 * math.log(q) * (1 - 2 * s)
            mat_size = np.asarray([0.0, 0.0, inner])
            csd[0][encode_csd([0, 0])] = 1
            for d in (1, 2):
                csd[d][encode_csd([0, 2])] = s
                csd[d][encode_csd([2, 0])] = s
                csd[d][encode_csd([1, 1])] = 1 - 2 * s
        elif self.shape_type == "112":
            s = self.split_0
            num_block = np.asarray([math.log(2), math.log(2), entropy([s, s, 1 - 2 * s])])
            mat_size = 2 * s * np.asarray([0.0, math.log(q), 0.0]) + (1 - 2 * s) * np.asarray([math.log(q), 0.0, math.log(q)])
            for d in (0, 1):
                csd[d][encode_csd([0, 1])] = 0.5
                csd[d][encode_csd([1, 0])] = 0.5
            csd[2][encode_csd([0, 2])] = s
            csd[2][encode_csd([2, 0])] = s
            csd[2][encode_csd([1, 1])] = 1 - 2 * s
        elif self.shape_type in ("013", "031"):
            num_block = np.zeros(3)
            mat_size = np.asarray([0.0, 0.0, math.log(2) + math.log(q)])
            csd[0][encode_csd([0, 0])] = 1
            if self.shape_type == "013":
                csd[1][encode_csd([0, 1])] = csd[1][encode_csd([1, 0])] = 0.5
                csd[2][encode_csd([1, 2])] = csd[2][encode_csd([2, 1])] = 0.5
            else:
                csd[1][encode_csd([1, 2])] = csd[1][encode_csd([2, 1])] = 0.5
                csd[2][encode_csd([0, 1])] = csd[2][encode_csd([1, 0])] = 0.5
        else:
            num_block = np.zeros(3)
            mat_size = np.zeros(3)
            csd[0][encode_csd([0, 0])] = 1
            csd[1][encode_csd([0, 0])] = 1
            csd[2][encode_csd([2, 2])] = 1
        self.num_block_contribution = np.asarray(rot3(list(num_block), self.rotate_num)) * self.term_frac
        self.mat_size_contribution = np.asarray(rot3(list(mat_size), self.rotate_num)) * self.term_frac
        self.complete_split = rot3(csd, self.rotate_num)


class TermInfoZero(BaseTerm):
    def __init__(self, model, level, term_id, shape, identifier):
        super().__init__(model, level, term_id, shape, identifier)
        if self.shape[0] == 0:
            self.zero_dim, self.nonzero_dims, self.base_mat_size = 0, (1, 2), np.asarray([0.0, 0.0, 1.0])
        elif self.shape[1] == 0:
            self.zero_dim, self.nonzero_dims, self.base_mat_size = 1, (0, 2), np.asarray([1.0, 0.0, 0.0])
        else:
            self.zero_dim, self.nonzero_dims, self.base_mat_size = 2, (0, 1), np.asarray([0.0, 1.0, 0.0])
        n = 3**self.power
        self.complete_split_id: list[list[Group | int]] = [[-1] * n for _ in range(3)]
        self.complete_split_id[self.zero_dim][0] = -2
        d1, d2 = self.nonzero_dims
        groups = []
        for idx in range(n):
            arr = decode_csd(idx, self.power)
            if int(arr.sum()) != int(self.shape[d1]):
                continue
            group = self.pm.register(1, 0, 1, f"zero[{level},{term_id}].csd[{idx}]")
            self.complete_split_id[d1][idx] = group
            opposite = encode_csd(2 - arr)
            self.complete_split_id[d2][opposite] = group
            groups.append((group, np.ones((1, 1))))
        self.pm.add_eq(groups, [1], f"zero[{level},{term_id}].csd-sum")
        self.num_block_contribution = [np.zeros(3) for _ in range(6)]
        self.hash_penalty_term = [0.0] * 6
        self.p_compY = [0.0] * 6
        self.p_compZ = [0.0] * 6

    def evaluate_init(self):
        super().evaluate_init()
        self.complete_split = []
        for ids in self.complete_split_id:
            csd = np.zeros(len(ids))
            for idx, group in enumerate(ids):
                if group == -2:
                    csd[idx] = 1
                elif group != -1:
                    csd[idx] = self.pm.get(group)[0]
            self.complete_split.append(csd)

    def evaluate_post(self):
        d1, _ = self.nonzero_dims
        inner = entropy(self.complete_split[d1])
        for idx, probability in enumerate(self.complete_split[d1]):
            inner += probability * int(np.sum(decode_csd(idx, self.power) == 1)) * math.log(self.model.q)
        self.mat_size_contribution = self.term_frac * inner * self.base_mat_size


class TermInfo(BaseTerm):
    def __init__(self, model, level, term_id, shape, identifier):
        super().__init__(model, level, term_id, shape, identifier)
        self.splits = prepare_splits(shape)
        self.num_split = len(self.splits)
        left = self.splits[:, :3]
        self.lam_low = left.min(axis=0)
        self.lam_high = left.max(axis=0)
        self.split_dist_id, self.split_dist_max_id = [], []
        for r in range(6):
            self.split_dist_id.append(self.pm.register(self.num_split, 0, 1, f"term[{level},{term_id}].r{r}.dist"))
            self.split_dist_max_id.append(self.pm.register(self.num_split, 0, 1, f"term[{level},{term_id}].r{r}.distmax"))
        self.region_prop_id = self.pm.register(6, 0, 1, f"term[{level},{term_id}].region")
        self.lam_margin_id, self.lam_sum_id = [], []
        for r in range(6):
            row = []
            for d in range(3):
                row.append(self.pm.register(int(self.lam_high[d] - self.lam_low[d] + 1), label=f"term[{level},{term_id}].r{r}.lam{d}"))
            self.lam_margin_id.append(row)
            self.lam_sum_id.append(self.pm.register(1, label=f"term[{level},{term_id}].r{r}.lamsum"))
        for r in range(6):
            self.pm.add_eq([(self.split_dist_id[r], np.ones((self.num_split, 1)))], [1], f"term[{level},{term_id}].r{r}.dist-sum")
            for d in range(3):
                size = self.sum_half + 1
                A = np.zeros((self.num_split, size))
                A[np.arange(self.num_split), self.splits[:, d]] = 1
                self.pm.add_eq([(self.split_dist_id[r], A), (self.split_dist_max_id[r], -A)], np.zeros(size), f"term[{level},{term_id}].r{r}.margin{d}")
        self.pm.add_eq([(self.region_prop_id, np.ones((6, 1)))], [1], f"term[{level},{term_id}].region-sum")

        self.left_ptr, self.right_ptr = [[] for _ in range(6)], [[] for _ in range(6)]
        for r in range(6):
            child_identifier = (self.term_id, r)
            for split in self.splits:
                self.left_ptr[r].append(model.find_or_create(level - 1, split[:3], child_identifier))
                self.right_ptr[r].append(model.find_or_create(level - 1, split[3:], child_identifier))

    def evaluate_init(self):
        super().evaluate_init()
        self.region_prop = self.pm.get(self.region_prop_id)
        self.split_dist = [self.pm.get(g) for g in self.split_dist_id]
        self.split_dist_max = [self.pm.get(g) for g in self.split_dist_max_id]
        self.lam_margin = [[self.pm.get(g) for g in row] for row in self.lam_margin_id]
        self.lam_sum = [self.pm.get(g)[0] for g in self.lam_sum_id]

    def evaluate_pre(self):
        for i in range(self.num_split):
            for r in range(6):
                amount = self.term_frac * self.split_dist[r][i] * self.region_prop[r]
                self.left_ptr[r][i].term_frac += amount
                self.right_ptr[r][i].term_frac += amount

    def evaluate_post(self):
        self.hash_penalty_term, self.num_block_contribution = [], []
        for r in range(6):
            self.hash_penalty_term.append((entropy(self.split_dist_max[r]) - entropy(self.split_dist[r])) * self.term_frac * self.region_prop[r])
            mar = margins(self.split_dist[r], self.splits[:, :3], self.sum_half)
            self.num_block_contribution.append(self.term_frac * self.region_prop[r] * np.asarray([entropy(v) for v in mar]))
        self.mat_size_contribution = np.zeros(3)
        self.complete_split_region = [[None] * 3 for _ in range(6)]
        for r in range(6):
            for d in range(3):
                total = np.zeros(3**self.power)
                for i in range(self.num_split):
                    total += self.split_dist[r][i] * np.kron(self.left_ptr[r][i].complete_split[d], self.right_ptr[r][i].complete_split[d])
                self.complete_split_region[r][d] = total
        self.complete_split = []
        for d in range(3):
            self.complete_split.append(sum(self.region_prop[r] * self.complete_split_region[r][d] for r in range(6)))
        self.p_compY = [(self._pcomp_num(r, 1) - self._pcomp_den(r, 1)) * self.term_frac * self.region_prop[r] for r in range(6)]
        self.p_compZ = [(self._pcomp_num(r, 2) - self._pcomp_den(r, 2)) * self.term_frac * self.region_prop[r] for r in range(6)]

    def _pcomp_num(self, region: int, which: int) -> float:
        dim_x, dim_y, dim_z = PERMS[region]
        target = dim_y if which == 1 else dim_z
        weighted = [np.zeros(3 ** (self.power // 2)) for _ in range(self.sum_half + 1)]
        probs = np.zeros(self.sum_half + 1)
        direct = 0.0
        for i in range(self.num_split):
            for ptr in (self.left_ptr[region][i], self.right_ptr[region][i]):
                if ptr.shape[target] == 0:
                    continue
                is_direct = ptr.shape[dim_z] == 0 if which == 1 else int(ptr.shape.min()) == 0
                if is_direct:
                    direct += self.split_dist[region][i] * entropy(ptr.complete_split[target])
                else:
                    key = int(ptr.shape[target])
                    weighted[key] += self.split_dist[region][i] * ptr.complete_split[target]
                    probs[key] += self.split_dist[region][i]
        return direct + sum(normalized_entropy(weighted[k], probs[k]) for k in range(1, self.sum_half + 1) if probs[k] > 0)

    def _pcomp_den(self, region: int, which: int) -> float:
        dim_x, dim_y, dim_z = PERMS[region]
        target = dim_y if which == 1 else dim_z
        mar = margins(self.split_dist[region], self.splits[:, :3], self.sum_half)
        return entropy(self.complete_split_region[region][target]) - entropy(mar[target])

    def lagrange_residuals(self) -> np.ndarray:
        values = []
        for r in range(6):
            for i, split in enumerate(self.splits):
                left = split[:3]
                exponent = self.lam_sum[r] - 1
                for d in range(3):
                    exponent += self.lam_margin[r][d][left[d] - self.lam_low[d]]
                values.append(math.exp(exponent) - self.split_dist_max[r][i])
        return np.asarray(values)


class GlobalStage:
    def __init__(self, model: "Model"):
        self.model, self.pm = model, model.pm
        self.level = model.max_level
        self.power = 2 ** (self.level - 1)
        self.sum_col = 2**self.level
        self.shapes = prepare_shapes(self.level)
        self.num_shape = len(self.shapes)
        self.region_prop_id = self.pm.register(6, 0, 1, "global.region")
        self.dist_id, self.dist_max_id = [], []
        for r in range(6):
            self.dist_id.append(self.pm.register(self.num_shape, 0, 1, f"global.r{r}.dist"))
            self.dist_max_id.append(self.pm.register(self.num_shape, 0, 1, f"global.r{r}.distmax"))
        self.lam_margin_id, self.lam_sum_id = [], []
        for r in range(6):
            self.lam_margin_id.append([self.pm.register(self.sum_col + 1, label=f"global.r{r}.lam{d}") for d in range(3)])
            self.lam_sum_id.append(self.pm.register(1, label=f"global.r{r}.lamsum"))
        self.pm.add_eq([(self.region_prop_id, np.ones((6, 1)))], [1], "global.region-sum")
        for r in range(6):
            self.pm.add_eq([(self.dist_id[r], np.ones((self.num_shape, 1)))], [1], f"global.r{r}.dist-sum")
            for d in range(3):
                A = np.zeros((self.num_shape, self.sum_col + 1))
                A[np.arange(self.num_shape), self.shapes[:, d]] = 1
                self.pm.add_eq([(self.dist_id[r], A), (self.dist_max_id[r], -A)], np.zeros(self.sum_col + 1), f"global.r{r}.margin{d}")
        # The official square verifier adds region == 1/6 componentwise.
        self.pm.add_eq([(self.region_prop_id, np.eye(6))], np.ones(6) / 6, "global.square-symmetry")

        self.term_ptr = [[] for _ in range(6)]
        for r in range(6):
            for shape in self.shapes:
                self.term_ptr[r].append(model.create_top(self.level, shape, (0, r)))

    def evaluate_init(self):
        self.region_prop = self.pm.get(self.region_prop_id)
        self.dist = [self.pm.get(g) for g in self.dist_id]
        self.dist_max = [self.pm.get(g) for g in self.dist_max_id]
        self.lam_margin = [[self.pm.get(g) for g in row] for row in self.lam_margin_id]
        self.lam_sum = [self.pm.get(g)[0] for g in self.lam_sum_id]

    def evaluate_pre(self):
        for r in range(6):
            for i, ptr in enumerate(self.term_ptr[r]):
                ptr.term_frac = self.dist[r][i] * self.region_prop[r]

    def evaluate_post(self):
        self.hash_penalty_term, self.num_block = [], []
        for r in range(6):
            self.hash_penalty_term.append((entropy(self.dist_max[r]) - entropy(self.dist[r])) * self.region_prop[r])
            mar = margins(self.dist[r], self.shapes, self.sum_col)
            self.num_block.append(np.asarray([entropy(v) for v in mar]) * self.region_prop[r])
        self.mat_size = np.zeros(3)
        for level in range(2, self.level + 1):
            for term in self.model.terms[level]:
                self.mat_size += term.mat_size_contribution
        self.complete_split = [[None] * 3 for _ in range(6)]
        for r in range(6):
            for d in range(3):
                self.complete_split[r][d] = sum(self.dist[r][i] * ptr.complete_split[d] for i, ptr in enumerate(self.term_ptr[r]))
        self.p_compY = [(self._pcomp_num(r, 1) - self._pcomp_den(r, 1)) * self.region_prop[r] for r in range(6)]
        self.p_compZ = [(self._pcomp_num(r, 2) - self._pcomp_den(r, 2)) * self.region_prop[r] for r in range(6)]

    def _pcomp_num(self, region: int, which: int) -> float:
        dim_x, dim_y, dim_z = PERMS[region]
        target = dim_y if which == 1 else dim_z
        weighted = [np.zeros(3**self.power) for _ in range(self.sum_col + 1)]
        probs = np.zeros(self.sum_col + 1)
        direct = 0.0
        for i, ptr in enumerate(self.term_ptr[region]):
            if ptr.shape[target] == 0:
                continue
            is_direct = ptr.shape[dim_z] == 0 if which == 1 else int(ptr.shape.min()) == 0
            if is_direct:
                direct += self.dist[region][i] * entropy(ptr.complete_split[target])
            else:
                key = int(ptr.shape[target])
                weighted[key] += self.dist[region][i] * ptr.complete_split[target]
                probs[key] += self.dist[region][i]
        return direct + sum(normalized_entropy(weighted[k], probs[k]) for k in range(1, self.sum_col + 1) if probs[k] > 0)

    def _pcomp_den(self, region: int, which: int) -> float:
        dim_x, dim_y, dim_z = PERMS[region]
        target = dim_y if which == 1 else dim_z
        mar = margins(self.dist[region], self.shapes, self.sum_col)
        return entropy(self.complete_split[region][target]) - entropy(mar[target])

    def lagrange_residuals(self) -> np.ndarray:
        values = []
        for r in range(6):
            for i, shape in enumerate(self.shapes):
                exponent = self.lam_sum[r] - 1
                for d in range(3):
                    exponent += self.lam_margin[r][d][shape[d]]
                values.append(math.exp(exponent) - self.dist_max[r][i])
        return np.asarray(values)


class Model:
    def __init__(self, q=5, max_level=3, K=1.0):
        self.q, self.max_level, self.K_fixed = q, max_level, K
        self.pm = ParameterManager()
        self.terms: dict[int, list[BaseTerm]] = {level: [] for level in range(1, max_level + 1)}
        self._term_index: dict[tuple, BaseTerm] = {}

        self.num_retain_comp_id = [[None] * (max_level + 1) for _ in range(6)]
        for r in range(6):
            for level in range(2, max_level + 1):
                self.num_retain_comp_id[r][level] = self.pm.register(1, 0, np.inf, f"aux.retain.r{r}.lv{level}")
        self.num_retain_glob_id = [self.pm.register(1, 0, np.inf, f"aux.global.r{r}") for r in range(6)]
        self.single_mat_size_id = self.pm.register(1, 0, np.inf, "aux.single-matrix")
        self.omega_id = self.pm.register(1, 0, np.inf, "omega")
        self.K_id = self.pm.register(1, K, K, "K")
        self.global_stage = GlobalStage(self)

    def _new_term(self, level, term_id, shape, identifier):
        if level == 2:
            return TermInfoLv2(self, level, term_id, shape, identifier)
        if min(shape) == 0:
            return TermInfoZero(self, level, term_id, shape, identifier)
        return TermInfo(self, level, term_id, shape, identifier)

    def create_top(self, level, shape, identifier):
        term_id = len(self.terms[level]) + 1
        term = self._new_term(level, term_id, shape, identifier)
        self.terms[level].append(term)
        # Top-level terms are intentionally never merged.
        return term

    def find_or_create(self, level, shape, identifier):
        key = (level, tuple(int(v) for v in shape), tuple(identifier))
        if key in self._term_index:
            return self._term_index[key]
        term_id = len(self.terms[level]) + 1
        term = self._new_term(level, term_id, shape, identifier)
        self.terms[level].append(term)
        self._term_index[key] = term
        return term

    def load(self, path: Path | str):
        x = np.asarray(loadmat(path, squeeze_me=True)["params"], dtype=float).ravel()
        if len(x) != self.pm.num_input:
            raise ValueError(f"certificate has {len(x)} parameters, model registered {self.pm.num_input}")
        self.pm.x = x.copy()

    def evaluate(self) -> dict:
        # Initialization and propagation.
        self.global_stage.evaluate_init()
        for level in range(1, self.max_level + 1):
            for term in self.terms[level]:
                term.evaluate_init()
        self.global_stage.evaluate_pre()
        for level in range(self.max_level, 0, -1):
            for term in self.terms[level]:
                term.evaluate_pre()
        for level in range(1, self.max_level + 1):
            for term in self.terms[level]:
                term.evaluate_post()
        self.global_stage.evaluate_post()

        ceq = [("global.lagrange", self.global_stage.lagrange_residuals())]
        for level in range(1, self.max_level + 1):
            for term in self.terms[level]:
                if isinstance(term, TermInfo):
                    ceq.append((f"term[{level},{term.term_id}].lagrange", term.lagrange_residuals()))

        inequalities: list[tuple[str, float]] = []  # convention value <= 0
        inferred_comp = [[0.0] * (self.max_level + 1) for _ in range(6)]
        for level in range(3, self.max_level + 1):
            for r in range(6):
                dim_x, dim_y, dim_z = PERMS[r]
                num_block = sum((t.num_block_contribution[r] for t in self.terms[level]), np.zeros(3))
                penalty = sum(t.hash_penalty_term[r] for t in self.terms[level])
                py = sum(t.p_compY[r] for t in self.terms[level])
                pz = sum(t.p_compZ[r] for t in self.terms[level])
                retain = self.pm.get(self.num_retain_comp_id[r][level])[0]
                candidates = []
                for d in range(3):
                    correction = penalty if d == dim_x else py if d == dim_y else pz
                    inequalities.append((f"retain.r{r}.lv{level}.d{d}", retain - num_block[d] + correction))
                    candidates.append(num_block[d] - correction)
                inferred_comp[r][level] = min(candidates)

        num_block2 = sum((t.num_block_contribution for t in self.terms[2]), np.zeros(3))
        retain2 = self.pm.get(self.num_retain_comp_id[0][2])[0]
        for d in range(3):
            inequalities.append((f"retain.symmetric.lv2.d{d}", retain2 - num_block2[d]))
        inferred_comp[0][2] = float(np.min(num_block2))
        for r in range(1, 6):
            ceq.append((f"retain.r{r}.lv2.zero", self.pm.get(self.num_retain_comp_id[r][2])))

        inferred_global = []
        for r in range(6):
            dim_x, dim_y, dim_z = PERMS[r]
            nb = self.global_stage.num_block[r]
            corrections = {dim_x: self.global_stage.hash_penalty_term[r], dim_y: self.global_stage.p_compY[r], dim_z: self.global_stage.p_compZ[r]}
            retain = self.pm.get(self.num_retain_glob_id[r])[0]
            candidates = []
            for d in range(3):
                inequalities.append((f"retain.global.r{r}.d{d}", retain - nb[d] + corrections[d]))
                candidates.append(nb[d] - corrections[d])
            inferred_global.append(min(candidates))

        K = self.pm.get(self.K_id)[0]
        mat_size = self.global_stage.mat_size
        single = self.pm.get(self.single_mat_size_id)[0]
        candidates = [mat_size[0], mat_size[1], mat_size[2] / K]
        for d, candidate in enumerate(candidates):
            inequalities.append((f"single-matrix.d{d}", single - candidate))
        inferred_single = min(candidates)

        omega = self.pm.get(self.omega_id)[0]
        value = single * omega
        value += sum(self.pm.get(group)[0] for group in self.num_retain_glob_id)
        for level in range(2, self.max_level + 1):
            value += sum(self.pm.get(self.num_retain_comp_id[r][level])[0] for r in range(6))
        target = math.log(self.q + 2) * 2 ** (self.max_level - 1)
        inequalities.append(("schonhage", target - value))

        # Best omega from the same distributions after setting all auxiliary
        # min variables to the exact limiting values.
        nonmult = sum(inferred_global) + sum(inferred_comp[r][level] for r in range(6) for level in range(2, self.max_level + 1))
        tightened_omega = (target - nonmult) / inferred_single

        linear = self.pm.linear_residuals()
        max_ineq = max(v for _, v in inequalities)
        max_eq = max(float(np.max(np.abs(v))) for _, v in ceq + linear)
        return {
            "omega": omega,
            "K": K,
            "target": target,
            "value": value,
            "tightened_omega": tightened_omega,
            "inferred_single": inferred_single,
            "inferred_global": inferred_global,
            "inferred_comp": inferred_comp,
            "inequalities": inequalities,
            "equalities": ceq,
            "linear_equalities": linear,
            "max_ineq": max_ineq,
            "max_eq": max_eq,
            "max_bound": self.pm.bound_violation(),
            "max_violation": max(0.0, max_ineq, max_eq, self.pm.bound_violation()),
        }


def print_report(model: Model, result: dict, top: int = 12):
    print(f"parameters       : {model.pm.num_input}")
    print("terms by level   : " + ", ".join(f"L{lv}={len(model.terms[lv])}" for lv in range(1, model.max_level + 1)))
    print(f"stored omega     : {result['omega']:.12f}")
    print(f"tightened omega  : {result['tightened_omega']:.12f}")
    print(f"certificate value: {result['value']:.12f}  target={result['target']:.12f}")
    print(f"max inequality   : {result['max_ineq']:.3e}")
    print(f"max equality     : {result['max_eq']:.3e}")
    print(f"max bound        : {result['max_bound']:.3e}")
    print(f"MAX VIOLATION    : {result['max_violation']:.3e}")
    print("\nMost active/violated inequalities (positive means violation):")
    for label, value in sorted(result["inequalities"], key=lambda item: item[1], reverse=True)[:top]:
        print(f"  {value:+.12e}  {label}")
    print("\nLargest equality residual blocks:")
    eq_blocks = result["equalities"] + result["linear_equalities"]
    ranked = sorted(eq_blocks, key=lambda item: float(np.max(np.abs(item[1]))), reverse=True)[:top]
    for label, values in ranked:
        print(f"  {np.max(np.abs(values)):.12e}  {label}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("certificate", type=Path)
    parser.add_argument("--q", type=int, default=5)
    parser.add_argument("--level", type=int, default=3)
    parser.add_argument("--K", type=float, default=1.0)
    parser.add_argument("--top", type=int, default=12)
    args = parser.parse_args()
    model = Model(args.q, args.level, args.K)
    model.load(args.certificate)
    result = model.evaluate()
    print_report(model, result, args.top)


if __name__ == "__main__":
    main()
