# Captain probe state

`captain_ctl.py` stores own-stack probe specs and home-zip restore snapshots here.

Important files:

- `<tag>.json` - baseline/candidate submit pair is fired and waiting for `probe-collect`.
- `<tag>.prehome.zip` - restore point for the home `submissions/submission.zip`.
- `<tag>.failed.json` - baseline or candidate submit failed; inspect before retrying.
- `<tag>.home_zip_block.json` - restore integrity failed or the snapshot was missing; block staging/new probes until fixed.
- `_done/` - completed probe specs after `probe-collect --apply`.

If a Captain session dies mid-probe, run:

```text
python Agenters/Submit_Captain/captain_ctl.py digest
python Agenters/Submit_Captain/captain_ctl.py probe-collect --apply
```

If there is a `.failed.json`, verify the home zip is restored from the matching `.prehome.zip` before retrying or starting a new probe.
`captain_ctl.py probe-collect --apply` now verifies the restored home-zip SHA equals the pre-probe snapshot SHA before it
marks packets `probe_ok`. Run probes from a main loop that can poll/block, or split them into explicit submit and collect
turns; do not rely on a one-shot subagent being re-invoked by a detached poller.
