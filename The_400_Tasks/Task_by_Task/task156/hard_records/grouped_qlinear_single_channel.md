# task156 grouped QLinear single-channel route

Date: 2026-07-13
Worker: bo_ng_cdx_worker2

## Live/team bar

- Authoritative bar: cost/profile_cost **977**.
- Source: `python tools/ng_current_bar.py --task 156 --json`; authoritative source `manifest`.
- Owner/sha: Boladi, `de09544e65761459dcbef871afca238d281618a26c6c349c224e48bad643586b`.

## Attempted route

Grouped `QLinearConv` with `groups=2` to shrink final weights from `[10,2,3,3]` to `[10,1,3,3]`.

## Failed pre-gate

Visible-window collision check over all 265 visible pairs. No ONNX was built because the representation cannot be a function of a single feature channel:

- feature channel 0 has conflicting identical 3x3 windows for colors 0, 1, 2, and 4; first color-0 conflict window `(0,0,0,0,1,1,0,1,1)` maps to labels `{0,1}`.
- feature channel 1 has conflicting identical 3x3 windows for colors 0, 2, and 4; first color-0 conflict window `(0,0,0,1,1,1,1,1,1)` maps to labels `{0,1}`.

## Why dead

With `groups=2`, output color channels 0-4 can read only one of the two feature channels. At least colors 0, 2, and 4 require information from both current feature channels on the visible corpus, so a grouped final bank would have a certain visible miss.

## Remaining hypothesis

Only a different two-channel feature encoding that moves all needed colors 0/1/2/4 separability into one grouped input lane could reopen grouped QLinearConv. The current affine feature encoding cannot.
