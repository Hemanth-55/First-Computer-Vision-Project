"""Clean the raw scoreboard reads into Hemanth's innings timeline (runs, balls) and the team score."""
import json, re, sys
from collections import Counter

def bat(s):
    s = (s or '').replace('|', ' ').replace('°', '').replace('.', ' ')
    if 'HEMANTH' not in s: return None
    m = re.search(r'HEMANTH\s+(\d{1,3})\s*\(\s*(\d{1,3})\s*\)', s)
    return (int(m.group(1)), int(m.group(2))) if m else None

def team(s):
    m = re.fullmatch(r'\s*(\d{1,3})\s*-\s*(\d)\s*', (s or '').replace(' 9-', '-').replace('9-1', '-1') if False else (s or ''))
    return (int(m.group(1)), int(m.group(2))) if m else None

rows = json.load(open(sys.argv[1]))
reads = []
for r in rows:
    if not r.get('bug'): continue
    h = bat(r.get('b2')) or bat(r.get('b1'))
    reads.append({'t': r['t'], 'hem': h, 'team': team(r.get('score')), 'over': r.get('over')})

# group consecutive reads into plateaus; keep values seen at least twice, never decreasing
states, last = [], (0, 0)
i = 0
while i < len(reads):
    j = i
    while j < len(reads) and reads[j]['t'] - reads[i]['t'] < 20 and (reads[j]['hem'] is None or reads[i]['hem'] is None or reads[j]['hem'][1] == (reads[i]['hem'] or (0, 0))[1]):
        j += 1
    grp = reads[i:max(j, i + 1)]
    hv = Counter(g['hem'] for g in grp if g['hem']).most_common(1)
    tv = Counter(g['team'] for g in grp if g['team']).most_common(1)
    single_ok = False
    if hv and hv[0][1] == 1 and tv and states and states[-1]['team']:
        # a value read only once is kept when the team total moved by exactly the same runs on one ball
        dr, db = hv[0][0][0] - last[0], hv[0][0][1] - last[1]
        single_ok = db == 1 and tv[0][0][0] - states[-1]['team'][0] == dr
    if hv and (hv[0][1] >= 2 or single_ok) and hv[0][0][0] >= last[0] and hv[0][0][1] >= last[1]:
        states.append({'t0': grp[0]['t'], 't1': grp[-1]['t'], 'runs': hv[0][0][0], 'balls': hv[0][0][1],
                       'team': list(tv[0][0]) if tv else None})
        last = hv[0][0]
    i = max(j, i + 1)

# merge identical neighbours
merged = []
for s in states:
    if merged and (merged[-1]['runs'], merged[-1]['balls']) == (s['runs'], s['balls']): merged[-1]['t1'] = s['t1']
    else: merged.append(s)
for a, b in zip(merged, merged[1:]):
    a['next_runs'], a['next_balls'] = b['runs'] - a['runs'], b['balls'] - a['balls']
json.dump(merged, open(sys.argv[2], 'w'), indent=1)
for s in merged: print(s)
