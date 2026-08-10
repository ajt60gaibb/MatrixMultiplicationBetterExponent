# A Sharper Numerical Certificate for the Matrix Multiplication Exponent

This repository contains the manuscript, certificate, and independent verification code for the
computer-assisted world-record candidate

\[
\omega < 2.371310.
\]

The underlying fourth-power parameter point has dual-safe exponent
`2.3713098963360215`; the deliberately slack serialized certificate stores
`2.3713099046652784`. Tensoring the fourth-power degeneration with itself gives an exact
eighth-power realization. The analytic reduction is exact, while the current numerical validation uses
binary64 rather than outward-rounded interval arithmetic. Accordingly, the manuscript describes this as
a record candidate pending external reproduction and interval closure.

## Repository layout

```text
omega-2.371310/
├── README.md
├── CITATION.cff
├── .gitignore
├── .github/workflows/verify.yml
├── paper/
│   ├── main.tex
│   ├── references.bib
│   ├── main.bbl
│   └── omega_2.371310.pdf
├── certificate/
│   ├── W1.00_2.371310_safe_final.mat
│   ├── p4_region_epigraph_stage4.npz
│   └── SHA256SUMS
├── verification/
│   ├── python_verifier.py
│   ├── dual_safe_verifier.py
│   ├── ci_check.py
│   ├── requirements.txt
│   └── *.json
├── optimization/
│   ├── p4_region_epigraph.py
│   ├── p4_lower_region_polish.py
│   ├── global_stage_opt.py
│   └── certify_p4_region_candidate.py
└── docs/
    ├── FINAL_STAGE4_AUDIT.md
    └── P4_REFINED_AUDIT.md
```

## Reproduce the certificate values

Use Python 3.12. The reference environment used NumPy 2.3.5 and SciPy 1.17.0.

```bash
python3 -m pip install -r verification/requirements.txt
(cd certificate && sha256sum -c SHA256SUMS)

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python3 verification/python_verifier.py \
  certificate/W1.00_2.371310_safe_final.mat --top 20

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python3 verification/dual_safe_verifier.py \
  certificate/W1.00_2.371310_safe_final.mat

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python3 verification/ci_check.py
```

Expected headline values:

| Quantity | Value |
|---|---:|
| Previous released certificate | `2.3713389005434182` |
| Fresh tightened evaluation | `2.3713098963342496` |
| Dual-safe evaluation | `2.3713098963360215` |
| Stored safe certificate | `2.3713099046652784` |
| Eighth-power product check | `2.3713098963357138` |

The certificate SHA-256 is
`65d78064ef3686f850558b3881446403afe9d4910d2d2e046c3dab92c390ff83`.

## Build the paper

```bash
cd paper
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

## Optimization provenance

The `optimization/` scripts expose the 42-inequality epigraph search and safe-certificate rebuild.
Re-running the complete discovery path requires the published source certificate
`W1.00_2.371339.mat`, available from the More Asymmetry authors' OSF archive:
<https://osf.io/mw5ak/?view_only=5769f03789354793b61e11aac4dd85dd>.

That upstream code and certificate are not redistributed here because the downloaded archive does not
state a software license. Place the upstream MAT file at `external/W1.00_2.371339.mat`, or pass its path
explicitly to `optimization/certify_p4_region_candidate.py`.

## Licensing

MIT for code and CC BY 4.0 for the paper.
