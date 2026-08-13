# Final audit: `CW_5^8` exponent below `2.371301`

## Result

The final joint structural point in `certificate/joint_global_region_epigraph_final.npz` has independently reconstructed fourth-power value

\[
\omega_{\rm tight}=2.3713007071793480.
\]

The safety-slack certificate `certificate/W1.00_2.371301_safe_final.mat` stores

\[
\omega_{\rm safe}=2.3713007080124780<2.371301.
\]

The certificate SHA-256 is `7d68936e6aab7fe4a76073e0036ee06cd8253dcd8d3ab9a0d5689362fdec98f3`.

The exact tensor square is a valid degeneration of `CW_5^8` with the same exponent. No correlated eighth-power improvement is needed for

\[
\boxed{\omega<2.371301}.
\]

## Independent numerical checks

- Ordinary tightened verifier: `2.3713007071793480`.
- Conservative entropy-dual verifier: `2.3713007071793486`.
- Independent manual dense forward pass: `2.3713007071793486`.
- Structural NPZ claim discrepancy: `4.00e-15`.
- Maximum certificate inequality residual: `-9.99999898e-11`.
- Maximum certificate equality residual: `4.66294e-15`.
- Maximum bound violation: `0`.
- Maximum optimized row/global simplex residual: `2.22e-16`.
- Maximum serialization normalization adjustment: `3.73e-13`.
- Maximum entropy-witness residual: `1.47e-14`.
- Dual-safe margin below `2.371301`: `2.92821e-7`.
- Dual-safe logarithmic margin at the stored exponent: `1.74680e-9`.
- Explicit 36-cell eighth-power exponent: `2.3713007071793486`.
- Product row/column marginal residual: `2.78e-17`.

The principal machine-readable reports are `verification/W1.00_2.371301_safe_final.audit.json`, `verification/dual_safe_2.371301_audit.json`, `verification/joint_2.371301_final_independent_audit.json`, and `verification/eighth_power_product_2.371301_audit.json`.

## Legal optimization axes

The structural search changes two families already registered independently by the More Asymmetry fourth-power theorem:

1. six global distributions over the 45 triples `(i,j,k)` with `i+j+k=8`; and
2. 126 local six-region distributions for interior level-three terms.

The six global region masses remain exactly `1/6`. No theorem requires the six global shape distributions to be equal or permutation-coupled. Local conditional split laws and complete-split policies remain fixed, and no excluded row support is activated.

The final certificate refreshes dependent maximum-entropy distributions and KKT multipliers. It also normalizes inherited conditional and boundary simplexes at the `1e-13` scale to remove old serialization residuals. Those are witness/canonicalization changes, not additional optimization axes.

## Reproduction

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python3 verification/python_verifier.py \
  certificate/W1.00_2.371301_safe_final.mat --top 20

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python3 verification/dual_safe_verifier.py \
  certificate/W1.00_2.371301_safe_final.mat --public-bound 2.371301

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python3 verification/joint_candidate_audit.py \
  certificate/W1.00_2.371310_safe_final.mat \
  certificate/W1.00_2.371301_safe_final.mat \
  certificate/joint_global_region_epigraph_final.npz

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python3 verification/eighth_power_product_audit.py \
  certificate/W1.00_2.371301_safe_final.mat
```
