# Historical row-only stage-4 audit (superseded)

> This document records the earlier `omega < 2.371310` row-only certificate.
> It is retained for provenance and is superseded by
> `docs/FINAL_STAGE4_AUDIT.md`, the joint global/local certificate, and the
> public statement `omega < 2.371301`.

## Certified exponent

The strongest audited safe certificate is

`work/cw8_optimizer/W1.00_2.371310_safe_final.mat`

and stores

```text
omega = 2.3713099046652784
```

The full dense minimum at the same structural point is

```text
omega_tight = 2.3713098963342496
```

The stored value is deliberately larger by `8.33103e-9`, because every
auxiliary minimum and the Schonhage inequality have explicit inward slack.
Relative to the supplied `2.3713389005434182` certificate, the safe
improvement is `2.8995878e-5`.

## Feasibility

A fresh reload through the full released Python verifier gives:

```text
maximum inequality residual  -9.9999997e-10
Schonhage residual            -2.0967486e-9
maximum equality residual      3.3131831e-11
maximum bound violation        0
```

The equality residual is inherited unchanged from the source certificate's
fixed maximum-entropy witnesses.  The epigraph solver's own auxiliary
variables are not part of the certificate: its direct row solution was
installed and all certificate auxiliaries were rebuilt from the fresh dense
minima.

## Independent dense forward audit

`verify_p4_dense_candidate.py` does not call `Model.evaluate` or
`CachedGlobalObjective`.  It explicitly drives each term's initialization,
pre-propagation, and post-propagation methods, then independently sums every
unminimized level-3, level-2, global, and matrix candidate and all certificate
inequalities.

Its maximum discrepancies from the epigraph optimizer's serialized candidate
arrays are:

```text
level 3              8.33e-17
level 2              7.77e-16
global compatibility 3.31e-12
matrix                7.11e-15
tightened omega       1.47e-12
```

The dense checker obtains `2.3713098963342496`, exactly matching the full
verifier at displayed precision.

## Parameter-difference audit

Comparing all 24,855 parameters with the supplied certificate finds exactly
141 changed parameter groups and 543 changed scalar coordinates:

- 126 independently registered level-3 `TermInfo.region_prop` simplexes;
- 14 auxiliary retain/global/single-matrix groups;
- the stored `omega` group.

There are no missing expected groups and no unexpected changed groups.  The
126 installed rows agree bit-for-bit with the final stage-4 candidate.  Their
maximum simplex residual is `2.22e-16`; no zero source-support coordinate was
activated; the smallest positive source-support weight is `2.38013e-7`.

Each interior `TermInfo` owns its own six-entry local-region simplex in the
released construction.  The rows are not tied across parents.  Varying them
only repartitions occurrences of the already revealed parent term among the
six legal hashing orientations.  Conditional split laws and maximum-entropy
witnesses stay fixed.  Lower contributions are affine in each row, while the
global compatibility calculation is recomputed from the resulting mixed
positional CSD.  Thus this axis is covered directly by the released theorem
and has none of the refined-label concerns of the experimental occurrence
menus.

## Explicit eighth-power lift

The refined 36-cell ordered-factor product lift is serialized in

`work/cw8_exact_model/refined_aggregate_product_epigraph_final.json`.

It gives

```text
CW^8 stored product objective 2.3713099036652783
CW^8 tightened product omega  2.3713098963357138
fourth-power tightened omega  2.3713098963342496
difference                    1.46e-12
left refined-marginal error   1.73e-17
right refined-marginal error  3.47e-17
```

The 36 orientation-cell masses are approximately `1/36`.  Inherited lower
retention and matrix terms are additive.  This is an explicit eighth-power
realization of the improved point, not an orbit-compressed surrogate.

## Reproduction

```bash
python3 work/more_asymmetry/python_verifier.py \
  work/cw8_optimizer/W1.00_2.371310_safe_final.mat --top 20

python3 work/cw8_exact_model/verify_p4_dense_candidate.py \
  work/more_asymmetry/data/W1.00_2.371339.mat \
  work/cw8_optimizer/W1.00_2.371310_safe_final.mat \
  work/cw8_optimizer/p4_region_epigraph_stage4.npz \
  --output work/cw8_exact_model/p4_dense_epigraph_final_independent_verification.json

python3 work/cw8_theory/refined_pair_model.py \
  work/cw8_optimizer/W1.00_2.371310_safe_final.mat --aggregate-theta \
  --output work/cw8_exact_model/refined_aggregate_product_epigraph_final.json
```

SHA-256:

```text
65d78064ef3686f850558b3881446403afe9d4910d2d2e046c3dab92c390ff83  W1.00_2.371310_safe_final.mat
260ae58be2e4716a246c20945ba803f69c47d97b88f6c3b06a0dfc67cb7f7260  p4_region_epigraph_stage4.npz
332b76962fe298116dc3ef5386c6989efbfd606dc4a2aa48a146d654a6a4b66d  p4_dense_epigraph_final_independent_verification.json
78499eb28defd3b797a7bfec0fac614707c88c1e6666ca13e75346fa07e93c60  refined_aggregate_product_epigraph_final.json
```
