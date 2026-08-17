route: U8 interior-detect + Where passthrough

Live bar: 270, sha256 `42d0ac9b0b331728a533bdf493b13f5b9cdd13811b4f3e3c8a9604c548530fa1`, source `python tools/ng_current_bar.py --task 98 --json` / team manifest, owner Vamshi.

Attempted route: re-price the prompt's U8 detect + final Where idea against the refreshed 270 direct-output Conv, not against stale 14,635/900 task state.

Exact failed gate: pre-build price gate. The current live graph has no F32 detect plane left to narrow. A correct Where formulation still needs an interior condition plane plus a channel-aware ch0 true branch; even a cropped `[1,1,18,19]` U8/BOOL condition is 342 bytes before params, already above the 270 live bar. Per the gate ladder, no visible/truth gate was run on a representation whose lower bound cannot beat live.

Why dead: this route was a win against the stale F32-detect graph, but the promoted one-node direct Conv already deleted the charged detect tensor and all intermediates.

Remaining hypothesis: combine the final-output passthrough idea with a zero-intermediate structured final op; ordinary Where cannot get below the current param-only 270 bar.
