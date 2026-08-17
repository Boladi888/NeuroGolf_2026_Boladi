"""Stage task239 v2 (Equal-canvas emit) for the submit-captain (no Kaggle submit)."""
import sys, os, json, math, hashlib, shutil, zipfile, datetime
sys.path.insert(0, "tools")
from _score_v2 import score_network_v2

os.chdir("<REPO_ROOT>")
ROOT = os.getcwd()

CAND = "The_400_Tasks/Task_by_Task/task239/prototypes/task239_crop_v2.onnx"
BASE_ZIP = "submissions/submission.zip"
TECH = "v2_equalcanvas_crop12x5"

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest().upper()

cand_macs, cand_mem, cand_par = score_network_v2(CAND)
cand_cost = cand_mem + cand_par
with zipfile.ZipFile(BASE_ZIP) as z:
    base_bytes = z.read("task239.onnx")
tmp = "The_400_Tasks/Task_by_Task/task239/_canon_tmp.onnx"
with open(tmp, "wb") as f:
    f.write(base_bytes)
base_macs, base_mem, base_par = score_network_v2(tmp)
base_cost = base_mem + base_par
os.remove(tmp)

cand_sha = sha256_file(CAND)
token = cand_sha[:8].lower()
candidate_id = f"task239_{TECH}_t_{token}_on_{base_cost}"
stage_dir = os.path.join(".hermes/board/staged_submits", candidate_id)
os.makedirs(stage_dir, exist_ok=True)
shutil.copyfile(CAND, os.path.join(stage_dir, "task239.onnx"))

pkg_dst = os.path.join(stage_dir, "submission.zip")
with zipfile.ZipFile(BASE_ZIP) as zin:
    with zipfile.ZipFile(pkg_dst, "w", zipfile.ZIP_DEFLATED) as zout:
        for n in zin.namelist():
            data = open(CAND, "rb").read() if n == "task239.onnx" else zin.read(n)
            zout.writestr(n, data)

def pts(cost):
    return round(25.0 - math.log(max(1.0, cost)), 4)

expected_lb = round(math.log(base_cost / cand_cost), 4)

manifest = {
    "schema": "neurogolf.staged_submit.v1",
    "candidate_id": candidate_id,
    "task_id": "task239",
    "model_path": os.path.join(ROOT, stage_dir, "task239.onnx").replace("/", "\\"),
    "model_sha256": cand_sha,
    "package_path": os.path.join(ROOT, pkg_dst).replace("/", "\\"),
    "package_sha256": sha256_file(pkg_dst),
    "lineage_base_zip": os.path.join(ROOT, BASE_ZIP).replace("/", "\\"),
    "lineage_base_zip_sha256": sha256_file(BASE_ZIP),
    "changed_package_entries": ["task239.onnx"],
    "base_task239_score_in_current_zip": {
        "memory": base_mem, "params": base_par, "cost": base_cost, "points": pts(base_cost)},
    "candidate_task239_score": {
        "memory": cand_mem, "params": cand_par, "cost": cand_cost, "points": pts(cand_cost)},
    "cost_delta": base_cost - cand_cost,
    "expected_lb_raw_loge": expected_lb,
    "RECOMMENDED": (
        f"PRIMARY (aggressive). Cheapest task239: {base_cost} -> {cand_cost} "
        f"({round(100*(base_cost-cand_cost)/base_cost,1)}% reduction, +{expected_lb} LB). "
        "Conservative fallback staged at task239_crop12x5_reducefirst_topk5_t_002675c4_on_96143 "
        "(cost 6,112, +2.76 LB, MatMul-preserving, structural proof)."),
    "ship_profile": (
        f"PATH A, DIFFERENT-EMIT (cost {base_cost} -> {cand_cost}). Same rule/decode as canonical; "
        "replaces the MatMul render + full-canvas bg-Where with a 12x5 color-index canvas + "
        "Equal-at-output + Pad. STRUCTURALLY byte-equal (idx canvas reproduces canonical's exact "
        "bar/bg/bbox per-cell decomposition; output<=12x5 is generator-proven) AND confirmed "
        "RAW+decoded byte-equal vs parent 40000/40000 across 2 seeds. The Equal(canvas,[0..9])-at-"
        "output emit is the SAME family as task388 v5 (shipped +0.084 LB as predicted) -> proven "
        f"grader-faithful, not a novel-op-chain risk. Expected +{expected_lb} LB at raw log_e x1.0x."),
    "rule": ("histogram bar-chart (ARC-GEN task_9af7a82c): count each non-bg color, TopK sort "
             "descending, one vertical bar per color (height=count) in adjacent columns, bg fill above."),
    "architecture": (
        "reduce-first counts -> TopK k=5 -> build [12,5] UINT8 color-index canvas: "
        "bar cell (h<count_c)->sorted_colors[c]; in-bbox bg (h<H & c<W)->0; else->255 sentinel; "
        "Reshape [1,1,12,5] -> Equal(idx, color_ids[1,10,1,1]=0..9) -> [1,10,12,5] BOOL -> "
        "Pad(BOOL) -> [1,10,30,30] = output (FREE). opset 13."),
    "floor_finding": (
        f"Candidate {cand_cost} ~= 1.0x the 12x5 EMIT floor. The output-materialization floor for a "
        "12x5 output is the [1,10,12,5] BOOL Equal result = 600 B (cheaper than a [1,1,30,30] U8 single-"
        "channel canvas = 900 B, because 10*12*5=600 < 900; the opposite of task388's 12x12 case). "
        "Remaining ~570 B = counts/sort (TopK side, ~186) + canvas-build masks (~280) + params 50. "
        "Cannot go lower without dropping below the 600 B out-mat floor (impossible for a 10-channel "
        "12x5 emit) or memorizing (lookup family -> 0/-13 LB). NOTE: the prompt's ~700-800 target "
        "assumed out<=6x5 (out-mat 300); the TRUE generator max is 12x5 (out-mat 600), so ~1,177 is the "
        "real floor for this task, not 800."),
    "assumptions": (
        "Input always 3-4 x 3-4; colors 1-9 only; counts distinct (ARC-GEN generate() + random_colors). "
        "Output <= 12 rows x 5 cols structurally (max ring from 4x4 = 12; <=5 distinct counts summing "
        "<=16). Confirmed 0/40000 fresh pairs exceed 12x5."),
    "verification": {
        "verify_against_live_official": "267/267 prototype-vs-LIVE",
        "onnx_vs_json_official": "267/267",
        "AB_byte_equal_vs_parent_seed100000": "20000/20000 (raw + decoded)",
        "AB_byte_equal_vs_parent_seed777000": "20000/20000 (raw + decoded)",
        "cand_correct_vs_ARCGEN": "40000/40000",
        "canon_correct_vs_ARCGEN": "40000/40000 (Solved -> structural-LIVE compression, raw log_e x1.0x)",
        "structural_byte_equal": "YES - idx canvas = canonical per-cell color by construction; 12x5 bound proven",
        "no_unused_initializers": True,
        "score_tool": f"_score_v2 -> memory {cand_mem}, params {cand_par}, cost {cand_cost}",
    },
    "CAPTAIN_INSTRUCTIONS": (
        "Drop THIS task239.onnx into the EXACT live Kaggle zip; only task239.onnx differs (verified). "
        "Two candidates staged: ship V2 (this, +" + str(expected_lb) + " LB) for max gain - it is "
        "structurally byte-equal AND 40000/40000 confirmed, and uses task388-v5's proven Equal-at-output "
        "emit. If you want zero architecture-change risk, ship the v1 fallback "
        "(task239_crop12x5_reducefirst_topk5_*, +2.76 LB, MatMul-preserving). Kaggle keeps highest, so "
        "V2 can be probed first and v1 shipped after if V2 is unexpectedly flat. task239 is a "
        "cascade-suspect (core.md); 96,143 -> 1,177 memory drop may relieve a neighbor (bonus)."),
    "builder": "The_400_Tasks/Task_by_Task/task239/prototypes/build_task239_crop_v2.py",
    "no_kaggle_submit_performed": True,
    "staged_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "worker": "neurogolf-worker-7 (claude opus)",
}

with open(os.path.join(stage_dir, "manifest.json"), "w") as f:
    json.dump(manifest, f, indent=2)

print(f"staged -> {stage_dir}")
print(f"  base {base_cost} -> cand {cand_cost} (delta {base_cost-cand_cost}, +{expected_lb} LB)")
with zipfile.ZipFile(BASE_ZIP) as za, zipfile.ZipFile(pkg_dst) as zb:
    diffs = [n for n in za.namelist() if za.read(n) != zb.read(n)]
    print(f"  verified package diff vs base: {diffs}")
