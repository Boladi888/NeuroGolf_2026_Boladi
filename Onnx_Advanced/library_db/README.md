# library_db — generated agent lookup layer

This folder is the SQLite-facing index over the Onnx_Advanced Markdown library.
The Markdown files remain canonical; the database is generated and cited back to
source lines.

Use it before opening large docs:

```powershell
python Onnx_Advanced/library_db/lib_query.py search "F32 input read floor"
python Onnx_Advanced/library_db/lib_query.py task 243
python Onnx_Advanced/library_db/lib_query.py op ReduceMax BOOL
python Onnx_Advanced/library_db/lib_query.py facts --task task370
python Onnx_Advanced/library_db/lib_query.py facts --status stale --limit 20
python Onnx_Advanced/library_db/lib_query.py status
```

Regenerate after library edits:

```powershell
python Onnx_Advanced/library_db/build_library_db.py
```

`lib_lint.py` checks freshness by comparing the DB's source manifest with the
current routed Markdown and DB scripts. If the DB is stale, rebuild it.

Design contract:

- Do not hand-edit `onnx_library.sqlite` or `build_manifest.json`.
- Store compact structured rows, not full duplicated Markdown prose.
- Every fact-like row has status, confidence, source path, and source line.
- Queries default to current facts. Use `--status stale`/`--status disputed`
  for librarian audits. Historical/retracted facts are retained only
  when source docs provide explicit `library-fact` blocks with a non-current
  status.
- Reviewed lane packets that are too granular for `techniques.md` can land as
  `library-fact` blocks in `neurogolf_lesson_facts.md`; rebuild the DB after edits.
- Mutable facts such as ORT/scorer behavior should use `mutable: true` and
  `reverify_after` in future `library-fact` blocks when a concrete recheck date
  is known.

Manual fact override block format for future librarian use. Source docs can use
triple-backtick or tilde fences; this README is skipped by the parser so the
example below is not indexed.

````text
~~~library-fact
id: ort.reducemax.bool.opset20
task_id: task371
fact_type: ort_op_dtype
subject: ReduceMax BOOL input behavior
claim: ReduceMax BOOL inputs are rejected by ORT opset 20 and should not be generated unless reverified.
status: stale
confidence: high
source_path: ort_reality.md
source_line: 20
~~~
````
