# Final audit: `CW_5^8` exponent below `2.371310`

## Result

The final structural point in
`work/cw8_optimizer/p4_region_epigraph_stage4.npz` has independently
reconstructed fourth-power value

\[
  \omega_{\rm tight}=2.3713098963342496.
\]

The safety-slack certificate
`work/cw8_optimizer/W1.00_2.371310_safe_final.mat` stores

\[
  \omega_{\rm safe}=2.3713099046652784<2.371310.
\]

Its SHA-256 is
`65d78064ef3686f850558b3881446403afe9d4910d2d2e046c3dab92c390ff83`.

The exact tensor square is a valid degeneration of `CW_5^8` and has the same
exponent.  The independent 36-cell product evaluator returns
`2.371309896335715`, differing from the factor reconstruction by only
`1.47e-12`.  Hence the certificate proves

\[
  \boxed{\omega<2.371310}.
\]

No correlated/refined-label improvement is needed for this conclusion.

## Independent numerical checks

- Original verifier after tightening: `2.3713098963342496`.
- Independent dense forward pass: `2.371309896335721`.
- Maximum component discrepancy: `3.31e-12`.
- Serialized row discrepancy: `0`.
- Maximum row-simplex residual: `2.22e-16`.
- Minimum region weight: `0`.
- Maximum certificate inequality residual: `-9.9999997e-10`.
- Maximum certificate equality residual: `3.31319e-11`, inherited from the
  source maximum-entropy witness.
- Maximum bound violation: `0`.
- Product-lift row/column marginal residual: `3.63e-14`.
- Final epigraph residual: exactly `0`; epigraph and direct objectives agree
  within `3e-15`.

The detailed machine-readable audit is
`work/cw8_theory/p4_2.371310_final_independent_audit.json`; the product-lift
audit is `work/cw8_theory/refined_pair_product36_p4final.json`.

## Why the changed parameters are legal

The only structural changes are the six-entry `region_prop` simplex rows of
the 126 interior order-three `TermInfo` objects.  The released program treats
each such row as an independent probability vector: its only constraints are
nonnegativity and sum one.  These weights repartition occurrences of an
already revealed parent term among six local hashing orientations; they do
not alter a conditional split law or a maximum-entropy witness.

For fixed witnesses, each level-three, level-two, and matrix candidate is
affine in these weights.  The parent positional CSD is also affine before
entropy is taken.  The audit therefore rebuilds the mixed positional CSDs,
recomputes the nonlinear global compatibility candidates, takes all released
minima, and finally rebuilds the auxiliary retain/single/omega variables with
`1e-9` safety slack.  A parameter diff finds no structural change outside
these 126 rows; the remaining 15 changed groups are exactly those rebuilt
auxiliaries.  Parent identifiers remain part of the child keys, so this
optimization introduces no hidden policy sharing or occurrence untying.

The complete derivation and the independent-forward formulas are in
`work/cw8_theory/THEORY_AUDIT.md` and
`work/cw8_theory/p4_epigraph_independent_audit.py`.
