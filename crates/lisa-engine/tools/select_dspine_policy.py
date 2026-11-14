"""Select a Metal BS1 SPS policy from completed dspine_compare reports.

This follows Tencent scripts/tune_sps.py's measured-throughput selection and
larger-width tie break. All candidates must cover the same prompts. Run GPU
measurements sequentially, with cooldowns and serial controls, before selection.
"""
import argparse
import collections
import json
import pathlib

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('reports', type=pathlib.Path, nargs='+')
parser.add_argument('--out', type=pathlib.Path, required=True, help='Selection evidence JSON')
parser.add_argument('--install', action='store_true', help='Install the selected local profile for Lisa UI')
args = parser.parse_args()
identity = profile_path = None
measurements = collections.defaultdict(lambda: collections.defaultdict(list))
for path in args.reports:
    report = json.loads(path.read_text())
    if 'all_tokens_match' not in report:
        raise ValueError(f'Incomplete or diagnostic report: {path}')
    if identity is None:
        identity = report['policy_identity']
        profile_path = pathlib.Path(report['policy_path'])
    if identity != report['policy_identity'] or profile_path != pathlib.Path(report['policy_path']):
        raise ValueError('Profiles differ in device, checkpoint or schedule identity')
    block = report['draft_block_size']
    if not isinstance(block, int) or not 3 <= block <= 16:
        raise ValueError('Invalid draft block')
    prompt = tuple(report['prompt_tokens'])
    if not any(run['mode'] == 'serial' for run in report['runs']):
        raise ValueError('Missing serial controls')
    for run in report['runs']:
        if run['mode'] != 'dspine':
            continue
        width = run['verify_width']
        if not isinstance(width, int) or not 2 <= width <= block:
            raise ValueError('Reports need explicit valid verification widths')
        if run['decode_s'] <= 0 or run['decode_tokens'] <= 0:
            raise ValueError('Invalid measured duration or token count')
        mode = report.get('ssm', 'replay')  # Earlier replay-only reports predate this field.
        if mode not in ('capture', 'replay'): raise ValueError('Unknown SSM policy')
        measurements[(mode, block, width)][prompt].append(run)
if not measurements:
    raise ValueError('No DSpine measurements')
prompts = set(next(iter(measurements.values())))
if any(set(rows) != prompts for rows in measurements.values()):
    raise ValueError('Every candidate must cover the same prompts')
# Weight each prompt equally, independent of the number of repeats. The
# harmonic mean corresponds to equal output-token workloads across prompts.
scores = []
for (mode, block, width), rows in sorted(measurements.items()):
    seconds_per_token = []
    for runs in rows.values():
        seconds_per_token.append(sum(r['decode_s'] for r in runs) / sum(r['decode_tokens'] for r in runs))
    tps = len(seconds_per_token) / sum(seconds_per_token)
    scores.append({'ssm': mode, 'draft_width': block, 'verify_width': width, 'tokens_per_second': tps})
best = max(scores, key=lambda row: (row['tokens_per_second'], row['verify_width'], row['draft_width']))
policy = {'format': 'lisa-dspine-sps-metal-v1', 'identity': identity,
          'draft_width': best['draft_width'], 'verify_width': best['verify_width'], 'ssm': best['ssm']}
evidence = {'policy': policy, 'candidates': scores, 'prompts': len(prompts),
            'reports': [str(p.resolve()) for p in args.reports]}
args.out.write_text(json.dumps(evidence, indent=2)+'\n')
if args.install:
    # Limit installation to the engine's local SPS cache. The engine also
    # validates the complete identity again before applying this profile.
    parent = pathlib.Path.home() / '.cache/lisa/dspine-sps'
    if profile_path.parent != parent or profile_path.suffix != '.json':
        raise ValueError('Unexpected policy installation path')
    parent.mkdir(parents=True, exist_ok=True)
    temporary = profile_path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(policy, indent=2)+'\n')
    temporary.replace(profile_path)
    print('Installed', profile_path)
print(json.dumps(best, indent=2))
