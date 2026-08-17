# Baselines, Prior Work, And Optional Leads

## Effective Rung At Pack Time

At authoring time on 2026-07-03:

- workbook plus manifest bar: `task374` = 1366, owner Vamshi;
- staged qscale-share candidate: cost 1365, memory 1310 + params 55;
- staged candidate sha256:
  `a4b47de87e8f317148931ead2f15f21c0b0479e1824b25f665dd674a230e5e35`;
- target for new work: strictly below 1365.

There are conflicting older local/home values in source notes. For this pack,
use the staged 1365 rung as the effective target unless a live local refresh
proves a newer lower rung.

## What The 1365 Candidate Did

The qscale-share candidate keeps the two-feature ConvInteger output architecture
and removes one redundant scalar initializer. It reuses `q_scale=0.2` for all
QLinearConv scales and multiplies row/column count kernels by 5 to preserve
integer outputs.

Known validation recorded in the bundled manifest:

- visible: 267/267 pass;
- truth full: 200000/200000;
- ab full: 200267/200267 byte equal;
- battery: pass;
- static and profile cost: 1365.

## Prior Wall That Should Not Over-Narrow The Search

The 1150-1250 route tried to pack the two feature planes into one before the
final ConvInteger. The bundled hard record argues this is impossible for a
single-channel linear ConvInteger decoder because it can separate only three
argmax outcomes, while the task needs four classes:

- background;
- longest;
- second-longest;
- remaining positive length.

This is a real wall for that decoder family. It is not a proof that every
sub-1365 formulation is impossible.

## Useful Optional Leads

- Avoid materializing padded 30x30 label tensors before the final output.
- Reconsider the final op or output convention rather than only the feature
  planes feeding ConvInteger.
- Look for a rule-first encoding of longest, second-longest, and remaining
  classes that avoids both a 900-byte label canvas and two 200-byte feature
  planes.
- Treat row/column descriptors as first-class objects; do not carry per-cell
  tensors until they are needed.
- If using the qscale family, name the exact scalar, tensor, or initializer you
  expect to delete before building.

## Known Dead Ends

- Do not rank by raw row or column occupancy count.
- Do not collapse the four output classes into one binary feature plane with a
  linear ConvInteger decoder.
- Do not stage a one-byte tie.
- Do not call visible-only success ready.
- Do not rely on `ConstantOfShape` all-zero tricks; local scoring rejected that
  route in prior risk probes.
