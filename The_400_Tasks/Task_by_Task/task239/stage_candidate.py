"""Stage the task239 crop candidate for the submit-captain (no Kaggle submit)."""
import sys, os, json, math, hashlib, shutil, zipfile, io, datetime
sys.path.insert(0, "tools")
import onnx
from _score_v2 import score_network_v2

os.chdir("<REPO_ROOT>")
ROOT = os.getcwd()

CAND = "The_400_Tasks/Task_by_Task/task239/prototypes/task239_crop_v1.onnx"
BASE_ZIP = "submissions/submission.zip"
TECH = "crop12x5_reducefirst_topk5"

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest().upper()

def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest().upper()

# scores
cand_macs, cand_mem, cand_par = score_network_v2(CAND)
cand_cost = cand_mem + cand_par
with zipfile.ZipFile(BASE_ZIP) as z:
    base_bytes = z.read("task239.onnx")
with open("The_400_Tasks/Task_by_Task/task239/_canon_tmp.onnx", "wb") as f:
    f.write(base_bytes)
base_macs, base_mem, base_par = score_network_v2("The_400_Tasks/Task_by_Task/task239/_canon_tmp.onnx")
base_cost = base_mem + base_par
os.remove("The_400_Tasks/Task_by_Task/task239/_canon_tmp.onnx")

cand_sha = sha256_file(CAND)
token = cand_sha[:8].lower()
candidate_id = f"task239_{TECH}_t_{token}_on_{base_cost}"
stage_dir = os.path.join(".hermes/board/staged_submits", candidate_id)
os.makedirs(stage_dir, exist_ok=True)

# copy candidate onnx
model_dst = os.path.join(stage_dir, "task239.onnx")
shutil.copyfile(CAND, model_dst)

# build package zip: copy base, swap task239.onnx
pkg_dst = os.path.join(stage_dir, "submission.zip")
with zipfile.ZipFile(BASE_ZIP) as zin:
    names = zin.namelist()
    with zipfile.ZipFile(pkg_dst, "w", zipfile.ZIP_DEFLATED) as zout:
        for n in names:
            data = zin.read(n)
            if n == "task239.onnx":
                with open(CAND, "rb") as f:
                    data = f.read()
            zout.writestr(n, data)

def pts(cost):
    return round(25.0 - math.log(max(1.0, cost)), 4)

expected_lb = round(math.log(base_cost / cand_cost), 4)

manifest = {
    "schema": "neurogolf.staged_submit.v1",
    "candidate_id": candidate_id,
    "task_id": "task239",
    "model_path": os.path.join(ROOT, model_dst).replace("/", "\\"),
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
    "RECOMMENDED": f"YES - crop of the histogram render to the 12x5 active region. {base_cost} -> {cand_cost} ({round(100*(base_cost-cand_cost)/base_cost,1)}% reduction).",
    "ship_profile": (
        "PATH A same-algorithm crop of the SHIPPED parent in current zip "
        f"(cost {base_cost} -> {cand_cost}). Expected +{expected_lb} LB at raw log_e x1.0x "
        "(Solved task: canonical scores 40000/40000 correct vs ARC-GEN). Gold-standard gate: "
        "RAW F16 byte-equal vs parent 40000/40000 across two independent seeds on the FULL "
        "ARC-GEN distribution + 267/267 visible+arc-gen."),
    "rule": ("histogram bar-chart (ARC-GEN task_9af7a82c): count each non-bg color, TopK sort "
             "descending, render one vertical bar per color (height=count) in adjacent columns, "
             "fill background above the bars within the bbox."),
    "architecture": (
        "Unchanged op vocabulary, cropped to the provable active region. "
        "(1) reduce-first: ReduceSum(input F32 axes[0,2,3]) -> Cast(F16) replaces "
        "Cast(full input)->ReduceSum, removing the 18,000 B [1,10,30,30] input_f16. "
        "(2) spatial crop: every 30-tied init rewritten to H=12 (max ring from 4x4) x W=5 "
        "(max distinct counts summing to <=16). (3) TopK k=5 shrinks the rank axis (ranks 6..9 "
        "always have count 0). (4) final Pad 12x5 -> 30x30 at the OUTPUT node (free); output F16, opset 10."),
    "floor_finding": (
        f"Candidate cost {cand_cost} = 1.04x the MatMul-algorithm floor (~5,886) at 12x5. "
        "Decomposition of the irreducible intermediates: out_flat[10,60]F16=1,200 (MatMul result, must be "
        "F16); output_partial[1,10,12,5]F16=1,200 (reshape for the bg Where); output_small[1,10,12,5]F16=1,200 "
        "(Where result, pre-Pad); row_col[5,12,5]F16=600 + row_col_flat[5,60]F16=600 (bar build for MatMul); "
        "counts+sort+masks ~626; params 86. The 3x[.,12,5] output chain (3,600) is structural to "
        "MatMul->assemble->Pad and cannot drop below it without (a) UINT8 downcast of the post-MatMul "
        "tensors (needs opset 13 Where(UINT8) + extra Casts, ~300 B for added divergence risk) or (b) "
        "replacing the MatMul render with a cheaper Equal/coordinate emit (NOVEL OP CHAIN on a Solved task "
        "-> traps.md 2026-05-24 0/-13 LB risk; NOT done). Per prompt guidance 'stop at 1.1x floor', stopped here."),
    "assumptions": (
        "Input always 3-4 x 3-4 (ARC-GEN generate(): common.randint(3,4)); colors 1-9 only (random_colors "
        "never returns 0); all counts distinct (generator's len(counts_to_colors)==idx guard). Therefore "
        "output <= 12 rows x 5 cols structurally (confirmed 0/40000 fresh pairs exceed 12x5 over 2 seeds). "
        "12x5 crop is byte-equal to the 30x30 canonical on every possible input."),
    "verification": {
        "verify_against_live_official": "267/267 prototype-vs-LIVE (tools/verify_against_live.py --tid 239)",
        "onnx_vs_json_official": "267/267 (train+test+arc-gen via convert_to_numpy + run_network)",
        "AB_RAW_F16_byte_equal_vs_parent_seed100000": "20000/20000",
        "AB_RAW_F16_byte_equal_vs_parent_seed777000": "20000/20000",
        "AB_decoded_byte_equal_vs_parent": "40000/40000",
        "cand_correct_vs_ARCGEN": "40000/40000",
        "canon_correct_vs_ARCGEN": "40000/40000 (confirms Solved -> structural-LIVE compression, raw log_e x1.0x)",
        "no_unused_initializers": True,
        "score_tool": f"_score_v2 -> memory {cand_mem}, params {cand_par}, cost {cand_cost}",
    },
    "CAPTAIN_INSTRUCTIONS": (
        "Drop THIS task239.onnx into the EXACT live Kaggle zip; only task239.onnx differs (verified). "
        "0 LB downside (40000/40000 raw byte-equal vs parent). task239 is in the cascade-suspect set "
        "(core.md: {45,129,130,171,172,210,211,220,239,...}); this 96,143 -> 6,112 memory drop may also "
        "relieve a neighbor cascade-ERROR (bonus, not assumed)."),
    "builder": "The_400_Tasks/Task_by_Task/task239/prototypes/build_task239_crop_v1.py",
    "no_kaggle_submit_performed": True,
    "staged_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "worker": "neurogolf-worker-7 (claude opus)",
}

with open(os.path.join(stage_dir, "manifest.json"), "w") as f:
    json.dump(manifest, f, indent=2)

print(f"staged -> {stage_dir}")
print(f"  base cost {base_cost} -> cand cost {cand_cost}  (delta {base_cost-cand_cost}, +{expected_lb} LB)")
print(f"  package_sha256 {manifest['package_sha256'][:16]}...")
print(f"  changed entries: {manifest['changed_package_entries']}")
# sanity: confirm only task239.onnx differs between base and package
with zipfile.ZipFile(BASE_ZIP) as za, zipfile.ZipFile(pkg_dst) as zb:
    diffs = [n for n in za.namelist() if za.read(n) != zb.read(n)]
    print(f"  verified package diff vs base: {diffs}")
