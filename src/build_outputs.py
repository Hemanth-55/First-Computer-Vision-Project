"""Combine scoreboard, shot and pose outputs into metrics.json, strokes.json and the website data."""
import warnings; warnings.filterwarnings("ignore", category=RuntimeWarning)
import itertools, json, sys
from pathlib import Path
import numpy as np

W = Path(sys.argv[1]); OUT = Path(sys.argv[2]); OUT.mkdir(parents=True, exist_ok=True)
timeline = json.load(open(W / 'timeline.json'))
shots = json.load(open(W / 'shots.json'))
raw = json.load(open(W / 'scoreboard_raw.json'))
strokes = json.load(open(W / 'strokes_raw.json'))
J = [0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]

# ---- innings points: ball-by-ball where the board moved by exactly one ball
pts, prev = [], {'runs': 0, 'balls': 0, 'team': [0, 0]}
for s in timeline:
    p = {'balls': s['balls'], 'runs': s['runs'], 'team': s['team'], 'video_t': s['t0']}
    if s['balls'] - prev['balls'] == 1 and prev['balls'] > 0:
        p['ball_runs'] = s['runs'] - prev['runs']
    pts.append(p); prev = s

def at(b): return max([p for p in pts if p['balls'] <= b], key=lambda p: p['balls'], default={'runs': 0})['runs']
cuts = [(0, 13, 'Fast start'), (13, 24, 'Rebuild'), (24, 31, 'Acceleration'), (31, 38, 'Finish of the footage')]
phases = [{'name': n, 'from': a + 1, 'to': b, 'balls': b - a, 'runs': at(b) - at(a),
           'strike_rate': round((at(b) - at(a)) / (b - a) * 100)} for a, b, n in cuts]
fours = [p['balls'] for p in pts if p.get('ball_runs') == 4]
sixes = [p['balls'] for p in pts if p.get('ball_runs') == 6]

# ---- strokes: keep the nine verified striker strokes, attach the ball and its outcome
verified = {1: None, 3: 13.0, 6: None, 10: None, 13: None, 16: None, 19: None, 24: None, 40: None}  # shot -> min contact time
sel = [s for s in strokes if s['shot'] in verified and s['contact_t'] >= (verified[s['shot']] or 0)]
for s in sel:
    before = [p for p in pts if p['video_t'] <= s['contact_t']]
    pre = before[-1] if before else None
    nxt = [p for p in pts if p['video_t'] > s['contact_t'] and pre and p['balls'] == pre['balls'] + 1]
    s['ball'] = pre['balls'] + 1 if pre else None
    s['runs'] = (nxt[0]['runs'] - pre['runs']) if nxt else None
    s['score_before'] = f"{pre['runs']} ({pre['balls']})" if pre else None

def pose_at(s, a, b):
    ts = np.array(s['skeleton']['times'])
    F = np.array([[np.nan if v is None else v for v in f] for f in s['skeleton']['frames']]).reshape(len(ts), 17, 2)
    P = np.nanmedian(F[(ts >= a) & (ts <= b)], axis=0)
    return (P - np.nanmean(P[[11, 12]], axis=0))[J]

def spread(ps):
    return float(np.mean([np.nanmean(np.linalg.norm(p - q, axis=1)) for p, q in itertools.combinations(ps, 2)]))

stance = [pose_at(s, -.9, -.5) for s in sel]; contact = [pose_at(s, -.03, .03) for s in sel]
consistency = {'stance_spread': round(spread(stance), 3), 'contact_spread': round(spread(contact), 3)}
consistency['ratio'] = round(consistency['contact_spread'] / consistency['stance_spread'], 1)

n_bug = sum(r['bug'] for r in raw)
metrics = {
    'video': {'seconds': 146.2, 'frames': 3976, 'resolution': '2388x1668', 'source': 'FanCode broadcast highlights, S.S. Rajan Trophy (TNCA inter-district T20)',
              'title_on_broadcast': '73 off 44! Hemanth Kumar G guides Thiruvallur to 177'},
    'pipeline': {'scoreboard_samples': len(raw), 'scoreboard_read': n_bug, 'camera_shots': len(shots),
                 'delivery_shots': sum(s['delivery_view'] for s in shots), 'strokes_tracked': len(sel), 'keypoints': 17},
    'innings': {'points': pts, 'phases': phases, 'fours': fours, 'sixes': sixes,
                'boundary_runs_confirmed': 4 * len(fours) + 6 * len(sixes), 'runs_in_footage': pts[-1]['runs'], 'balls_in_footage': pts[-1]['balls'],
                'final': {'runs': 73, 'balls': 44, 'team_total': 177, 'source': 'broadcast video title'}},
    'consistency': consistency,
    'strokes': [{k: s[k] for k in ('shot', 'ball', 'runs', 'score_before', 'contact_t', 'height_px')} | {'skeleton': s['skeleton']} for s in sel],
}
json.dump(metrics, open(OUT / 'metrics.json', 'w'), indent=1)
print(json.dumps({k: v for k, v in metrics.items() if k not in ('strokes', 'innings')}, indent=1))
print('phases', phases); print('fours', fours, 'sixes', sixes)
for s in sel: print('stroke shot', s['shot'], 'ball', s['ball'], 'runs', s['runs'], s['score_before'])
