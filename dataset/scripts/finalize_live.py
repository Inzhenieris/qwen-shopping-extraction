#!/usr/bin/env python3
"""Wait for live shards, merge, verify against the seeded scenario source, and freeze splits."""
import argparse, hashlib, json, pathlib, sys, time

p = argparse.ArgumentParser()
p.add_argument('--ts', required=True)
p.add_argument('--wait-seconds', type=int, default=2700)
a = p.parse_args()
root = pathlib.Path(__file__).resolve().parents[1]
os_root = root
shards = sorted((root / 'data/generated').glob(f'live_{a.ts}_*.jsonl'))
logs = sorted((root / 'logs').glob(f'live_{a.ts}_*.log'))
if len(shards) != 10 or len(logs) != 10:
    print(f'Expected 10 shards/logs, found {len(shards)}/{len(logs)}', file=sys.stderr)
    sys.exit(1)

deadline = time.time() + a.wait_seconds
while time.time() < deadline:
    finished = 0
    for log in logs:
        text = log.read_text(errors='replace') if log.exists() else ''
        if 'Success: saved 250/250' in text:
            finished += 1
    print(f'finished shards: {finished}/10', flush=True)
    if finished == 10:
        break
    time.sleep(15)
else:
    print('Timed out waiting for shards', file=sys.stderr)
    sys.exit(1)

rows = []
for path in shards:
    for line in path.read_text().splitlines():
        if line.strip():
            rows.append(json.loads(line))
rows.sort(key=lambda r: r['id'])
src_path = root / 'data/scenarios/scenarios_2500_seed42.jsonl'
src = [json.loads(l) for l in src_path.read_text().splitlines() if l.strip()]
if len(rows) != 2500:
    print(f'Expected 2500 rows, found {len(rows)}', file=sys.stderr)
    sys.exit(1)
for row, expected in zip(rows, src):
    if row['id'] != expected['id'] or row['scenario'] != expected['scenario'] or row['gold'] != expected['gold']:
        print(f"Scenario/gold mismatch at id {row.get('id')}", file=sys.stderr)
        sys.exit(1)
    if not isinstance(row.get('sentence'), str) or not row['sentence'].strip():
        print(f"Missing sentence at id {row.get('id')}", file=sys.stderr)
        sys.exit(1)

out = root / 'data/generated/dataset_2500_live.jsonl'
if out.exists():
    print(f'{out} already exists; refusing to overwrite', file=sys.stderr)
    sys.exit(1)
out.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows))

split_dir = root / 'data/splits_live'
split_dir.mkdir(exist_ok=True)
defs = {'train': rows[:2000], 'val': rows[2000:2200], 'test': rows[2200:2500]}
manifest = {
    'source_scenarios': str(src_path),
    'source_scenarios_sha256': hashlib.sha256(src_path.read_bytes()).hexdigest(),
    'merged': str(out),
    'merged_sha256': hashlib.sha256(out.read_bytes()).hexdigest(),
    'total': len(rows),
    'counts_by_kind': {},
    'splits': {},
}
from collections import Counter
manifest['counts_by_kind'] = dict(Counter(r['scenario']['kind'] for r in rows))
for name, part in defs.items():
    sp = split_dir / f'{name}.jsonl'
    if sp.exists():
        print(f'{sp} already exists; refusing to overwrite', file=sys.stderr)
        sys.exit(1)
    sp.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in part))
    manifest['splits'][name] = {
        'path': str(sp), 'count': len(part),
        'first_id': part[0]['id'], 'last_id': part[-1]['id'],
        'sha256': hashlib.sha256(sp.read_bytes()).hexdigest(),
        'counts_by_kind': dict(Counter(r['scenario']['kind'] for r in part)),
    }
manifest['review_status'] = 'unreviewed; sentences require semantic review before training'
(split_dir / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
print(json.dumps(manifest, indent=2))
