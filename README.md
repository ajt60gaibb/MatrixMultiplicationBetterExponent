# A Sharper Numerical Certificate for the Matrix Multiplication Exponent

This repository contains the manuscript, certificate, optimization code, and independent audits for the computer-assisted world-record candidate

\[
\omega < 2.371301.
\]

The jointly optimized fourth-power point has ordinary exponent `2.3713007071793480` and conservative dual-safe exponent `2.3713007071793486`. The deliberately slack serialized certificate stores `2.3713007080124780`. Tensoring the fourth-power degeneration with itself gives an exact eighth-power realization.

The analytic reduction is exact, while the current numerical validation uses binary64 rather than outward-rounded interval arithmetic. The manuscript therefore describes the result as a record candidate pending external reproduction and interval closure.

## Repository layout

```text
├── paper/
│   ├── main.tex
│   ├── references.bib
│   └── omega_2.371301.pdf
├── certificate/
│   ├── W1.00_2.371301_safe_final.mat
│   ├── joint_global_region_epigraph_final.npz
│   └── SHA256SUMS
├── optimization/
│   ├── joint_epigraph.py
│   ├── joint_global_epigraph.py
│   ├── certify_joint_candidate.py
│   └── legacy row/global search scripts
├── verification/
│   ├── python_verifier.py
│   ├── dual_safe_verifier.py
│   ├── joint_candidate_audit.py
│   ├── eighth_power_product_audit.py
│   ├── ci_check.py
│   └── generated JSON audits
└── docs/
    ├── FINAL_STAGE4_AUDIT.md
    └── P4_REFINED_AUDIT.md
```

The older `2.371310` certificate and row-only artifact remain in the repository as historical provenance.

## Reproduce the certificate values

Use Python 3.12 or newer. The reference environment used NumPy 2.3.5 and SciPy 1.17.0.

```bash
python3 -m pip install -r verification/requirements.txt
(cd certificate && shasum -a 256 -c SHA256SUMS)

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python3 verification/python_verifier.py \
  certificate/W1.00_2.371301_safe_final.mat --top 20

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python3 verification/dual_safe_verifier.py \
  certificate/W1.00_2.371301_safe_final.mat --public-bound 2.371301

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python3 verification/ci_check.py
```

Expected headline values:

| Quantity | Value |
|---|---:|
| Previous released certificate | `2.3713389005434182` |
| Fresh tightened evaluation | `2.3713007071793480` |
| Dual-safe evaluation | `2.3713007071793486` |
| Stored safe certificate | `2.3713007080124780` |
| Maximum inequality residual | `-9.99999898e-11` |
| Maximum equality residual | `4.66293670e-15` |

Certificate SHA-256: `7d68936e6aab7fe4a76073e0036ee06cd8253dcd8d3ab9a0d5689362fdec98f3`.

## Optimization and certification

The improvement jointly releases exactly the structural variables permitted by the existing More Asymmetry program: six independent 45-cell global shape distributions and 126 independent six-region constituent distributions. The six global region masses remain fixed at `1/6`; conditional split laws and local complete-split policies remain fixed.

`optimization/joint_epigraph.py` represents all nested minima by 14 epigraph variables and 42 smooth inequalities. It records the best exact direct structural point seen, checks analytic directional derivatives, and verifies that point through a fresh full model. The output NPZ is then converted into a standalone, safety-slack MAT certificate by `optimization/certify_joint_candidate.py`, which reconstructs all 762 maximum-entropy primal/dual witnesses and removes SciPy's wall-clock MAT-header timestamp for byte-for-byte reproducibility.

To audit the published structural artifact independently:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python3 verification/joint_candidate_audit.py \
  certificate/W1.00_2.371310_safe_final.mat \
  certificate/W1.00_2.371301_safe_final.mat \
  certificate/joint_global_region_epigraph_final.npz

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python3 verification/eighth_power_product_audit.py \
  certificate/W1.00_2.371301_safe_final.mat
```

The upstream `2.371339` source certificate is available from the More Asymmetry authors' [OSF archive](https://osf.io/mw5ak/?view_only=5769f03789354793b61e11aac4dd85dd). Their upstream code and certificate are not redistributed here because the downloaded archive does not state a software license.

## Build the paper

```bash
cd paper
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

## Licensing

MIT for code and CC BY 4.0 for the paper.
