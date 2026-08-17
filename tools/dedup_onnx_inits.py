"""Fast version: just dedupe and write zip, skip scoring."""
import sys, struct, pathlib, zipfile, os
os.environ['ORT_LOGGING_LEVEL'] = '3'
import onnx
from collections import OrderedDict


def init_data_tuple(init):
    if init.data_type == 7:
        if init.raw_data:
            n = len(init.raw_data) // 8
            return tuple(struct.unpack('<' + 'q' * n, init.raw_data))
        return tuple(init.int64_data)
    elif init.data_type == 1:
        if init.raw_data:
            n = len(init.raw_data) // 4
            return tuple(struct.unpack('<' + 'f' * n, init.raw_data))
        return tuple(init.float_data)
    elif init.data_type == 6:
        if init.raw_data:
            n = len(init.raw_data) // 4
            return tuple(struct.unpack('<' + 'i' * n, init.raw_data))
        return tuple(init.int32_data)
    return None


def dedupe(model):
    seen = OrderedDict()
    rename = {}
    keep_inits = []
    for init in model.graph.initializer:
        d = init_data_tuple(init)
        if d is None:
            keep_inits.append(init)
            continue
        key = (tuple(init.dims), init.data_type, d)
        if key in seen:
            rename[init.name] = seen[key]
        else:
            seen[key] = init.name
            keep_inits.append(init)
    for node in model.graph.node:
        for i, inp_name in enumerate(node.input):
            if inp_name in rename:
                node.input[i] = rename[inp_name]
    del model.graph.initializer[:]
    model.graph.initializer.extend(keep_inits)
    return model, len(rename)


def main():
    repo = pathlib.Path(__file__).resolve().parents[1]
    NG = pathlib.Path(os.environ.get('NEUROGOLF_NOGIT_ROOT', repo / '.notongit' / 'NeuroGolf_nogit'))
    src_mega = NG / 'submissions/_alpha_a_scratch/alpha_a_mega_t089v2_t137v2/submission.zip'
    new_dir = NG / 'submissions/_alpha_a_scratch/alpha_a_mega_t089v2_t137v2_dedup'
    new_dir.mkdir(exist_ok=True)
    new_zip = new_dir / 'submission.zip'

    n_total_renamed = 0
    n_failed = 0
    n_processed = 0

    new_entries = {}
    with zipfile.ZipFile(src_mega) as zin:
        names = sorted(zin.namelist())
        for name in names:
            data = zin.read(name)
            if not name.endswith('.onnx'):
                new_entries[name] = data
                continue
            try:
                m = onnx.load_from_string(data)
                m_new, n_renamed = dedupe(m)
                new_data = m_new.SerializeToString()
                new_entries[name] = new_data
                n_total_renamed += n_renamed
            except Exception as e:
                new_entries[name] = data  # keep original
                n_failed += 1
            n_processed += 1

    print(f'Processed {n_processed} ONNX files, {n_failed} failed (kept original)')
    print(f'Total inits deduplicated: {n_total_renamed}')

    with zipfile.ZipFile(new_zip, 'w', zipfile.ZIP_DEFLATED) as zout:
        for name, data in sorted(new_entries.items()):
            zout.writestr(name, data)

    print(f'Wrote {new_zip} ({new_zip.stat().st_size} bytes)')


if __name__ == '__main__':
    main()
