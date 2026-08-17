# GUI Task Payloads

`task*.json` files are intentionally not tracked here. They duplicate the
competition data under `data/competitions/neurogolf-2026/`, which is supplied by
the local `data/` junction outside git.

When the FastAPI server is used, `/tasks/taskNNN.json` is served from the built
client, this directory, `NEUROGOLF_DATA_DIR`, or the project data directory, in
that order.

For a pure static/Vite workflow, regenerate local copies before launching:

```powershell
New-Item -ItemType Directory -Force GUI/client/public/tasks | Out-Null
Copy-Item data/competitions/neurogolf-2026/task*.json GUI/client/public/tasks/
```
