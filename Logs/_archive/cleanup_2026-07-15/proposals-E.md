# Agent E Proposals

## Consolidate the two `.update` archives after Agent B finishes provenance repair

Proposed text for the owner or Agent B to fold into the update/provenance cleanup:

The root `.update/` inbox is fully drained: all 46 tracked files live under `_archive/update_integrated_2026-07/`, and there are zero pending root inbox files. `Onnx_Advanced/.update/` is a second drained inbox with no filename overlap against the root archive. Because this split already caused a dead-provenance bug, consolidate the root archive under the `Onnx_Advanced/.update/` system after the Agent B provenance fix lands.

Recommended shape:

- Move the 46 root `.update/_archive/update_integrated_2026-07/` files into a clearly named subfolder under `Onnx_Advanced/.update/_archive/`, such as `root_update_integrated_2026-07/`.
- Add one short provenance note in the surviving `Onnx_Advanced/.update/` README/catalog explaining that the moved files came from the old root `.update` inbox.
- Remove the top-level `.update/` folder after the move so future agents have one update inbox to inspect.

Do not flatten the two archives together without the provenance note; the lack of a single obvious update source is exactly what made the previous reference bug easy to miss.
