# Algorithm_Library — the algorithm pattern library

The hardest part of optimizing a NeuroGolf task isn't writing ONNX. It's figuring out **what the task computes**. Once you know the rule, the ONNX is mechanical.

Read in this order:

1. `rule_discovery_methodology.md` — how to figure out a task's rule from its train/test/arc-gen pairs. Visual inspection patterns, cross-pair consistency checks, when to trust visible-only verification.
2. `algorithm_patterns.md` — recurring ARC algorithms with minimal-cost ONNX templates. Bounding box, color permutation, connected components, periodic-fill, etc. Cite the tasks that used each.
3. `per_family_starting_points.md` — quick-reference for each of the four task families. Which techniques to try first.

After this folder, return to `Onnx_Advanced/task_techniques_index.md` to find your specific task entry.
