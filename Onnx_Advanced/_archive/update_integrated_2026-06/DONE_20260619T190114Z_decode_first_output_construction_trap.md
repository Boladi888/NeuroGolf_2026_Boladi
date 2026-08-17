# Decode-first trap: read the generator's output-construction branch, not just its draw order

Source: task366 worker session 2026-06-19. Transferable to any ARC-GEN task whose
generator builds `input` (grid) and `output` in the SAME loop.

## The trap
A prompt (and a prior agent) claimed task366's output boxes are "drawn DIRECTLY"
from one input panel, so the expensive library-match machinery could be dissolved
into a #118 crop + #109/#119 free-output Einsum. This came from skimming the
generator's draw loop and seeing `output[r][c] = forecolor` over a box rectangle.

The actual loop (`external/ARC-GEN/tasks/task_e6721834.py` lines 96-115) writes the
forecolor RECTANGLE into the input `grid` ONLY in the `else` branch
(`bside == foreside`), while the cross boxes that form the output get only their
DOTS written into the grid. So the input panel that the output is built from
contains NO solid rectangle — only sparse dots. The rectangle EXTENT lives only in
the *other* panel (the library), and recovering it requires matching the dot-cluster
to the library box by shared sprite index.

## The rule
When a generator writes both `input` and `output` inside one loop with a
`bside != foreside` (or any "this element goes to output vs input") conditional,
trace EXACTLY which writes land in `input` vs `output` per branch. A field that
appears in `output[...] = X` is NOT necessarily present in any input panel. Drawing
the output "directly" from an input crop is only valid if every output cell's value
AND extent is materially present in that crop.

## Cheap falsification (do this before building any free-output candidate)
For ~400 random `generate()` samples, compute the bounding box of the output's
non-bg/non-forecolor "marker" cells and the bounding box of the forecolor region.
If forecolor extent exceeds the marker bbox in a large fraction of cases, the
extent is NOT recoverable from markers alone and a marker-crop Einsum is
information-insufficient (task366: 380/400). This is a 20-line numpy probe that
kills a multi-thousand-line build before it starts.

## Cross-link
Reinforces task366's existing sub-1k floor proof (V24-V62, 2026-06-05): no static
free-output route is hidden-correct. This note supplies the upstream reason — the
information isn't in the half being cropped.
