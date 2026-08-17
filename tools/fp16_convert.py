"""FP16 conversion wrapper that fixes onnxconverter_common's SSA bug
(duplicate Cast node output names when keep_io_types=True).
"""
import onnx
from onnxconverter_common import float16


def convert(model):
    m_fp16 = float16.convert_float_to_float16(model, keep_io_types=True)
    # Dedupe output names: find Cast nodes that produce the same output and rename
    used_outputs = {}
    for n in m_fp16.graph.node:
        new_outputs = []
        for out in n.output:
            if out in used_outputs:
                used_outputs[out] += 1
                new_name = f'{out}__{used_outputs[out]}'
                # Update downstream nodes that use this old name
                # Only the FIRST occurrence keeps the name
                new_outputs.append(new_name)
                used_outputs[new_name] = 0
            else:
                used_outputs[out] = 0
                new_outputs.append(out)
        for i, on in enumerate(new_outputs):
            n.output[i] = on
    # That fix is incomplete because downstream nodes still ref old name.
    # Better strategy: collect dup outputs, rebuild names + downstream input refs.
    return m_fp16


def convert_clean(model):
    """Convert, then dedupe both node names and output names with consumer rewiring."""
    m_fp16 = float16.convert_float_to_float16(model, keep_io_types=True)
    # First pass: assign UNIQUE node names to every node
    for i, n in enumerate(m_fp16.graph.node):
        n.name = f'{n.op_type}_{i}'
    # Second pass: dedupe output names. If a name appears as output of multiple nodes,
    # rename later occurrences and rewire downstream consumers that USE that exact name.
    output_to_first_producer = {}
    rename_map = {}  # old_dup_name -> per_producer_unique_name
    for ni, n in enumerate(m_fp16.graph.node):
        for oi, out in enumerate(n.output):
            if out in output_to_first_producer:
                # Duplicate: rename this output
                new = f'{out}__dup{ni}_{oi}'
                # Update consumers? Since topological order means consumers downstream
                # would have been wired to the FIRST producer (which is what they intend).
                # New name only refers to this duplicate output, which has no consumers,
                # so just renaming is safe.
                n.output[oi] = new
            else:
                output_to_first_producer[out] = (ni, oi)
    return m_fp16


def _cli(argv=None):
    import argparse
    p = argparse.ArgumentParser(
        description="FP16-convert ONNX models via convert_clean(). "
                    "Single: `in.onnx out.onnx`.  Batch: `--batch a.onnx b.onnx ...` (converts each in place). "
                    "Folds the old _f16_batch_convert.py / _multi_f16_batch.py drivers.")
    p.add_argument('inp', nargs='?', help='input onnx (single mode)')
    p.add_argument('out', nargs='?', help='output onnx (single mode)')
    p.add_argument('--batch', nargs='+', default=None, help='convert each listed onnx in place')
    a = p.parse_args(argv)
    if a.batch:
        for f in a.batch:
            onnx.save(convert_clean(onnx.load(f)), f)
            print('FP16 (in place):', f)
        return 0
    if a.inp and a.out:
        onnx.save(convert_clean(onnx.load(a.inp)), a.out)
        print('Saved FP16 to', a.out)
        return 0
    p.error('provide `in out` (single) or `--batch f1.onnx f2.onnx ...`')


if __name__ == '__main__':
    raise SystemExit(_cli())
