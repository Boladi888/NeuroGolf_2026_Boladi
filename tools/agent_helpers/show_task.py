#!/usr/bin/env python
"""show_task.py --task N  — print ONE task's row from task_state.csv (1 MB; NEVER Read it whole)."""
import csv, argparse, os, sys
ap = argparse.ArgumentParser(); ap.add_argument('--task', required=True); a = ap.parse_args()
key = str(int(a.task)).zfill(3)
root = os.path.join(os.path.dirname(__file__), '..', '..')
with open(os.path.join(root, 'The_400_Tasks', 'task_state.csv'), encoding='utf-8', newline='') as f:
    r = csv.DictReader(f)
    for row in r:
        tid = (row.get('task') or row.get('task_id') or '').replace('task','').zfill(3)
        if tid == key:
            for k, v in row.items():
                if v and len(str(v)) < 400: print(f'{k}: {v}')
            sys.exit(0)
print(f'task{key}: no row found'); sys.exit(1)
