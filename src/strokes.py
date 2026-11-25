"""Find batting strokes inside the striker tracks and measure them.

A stroke = a continuous track of the striker (>= 0.45 s) whose hands reach a clear speed peak.
All distances are in units of the batter's own height (head to ankles), so camera zoom doesn't matter.
"""
import warnings; warnings.filterwarnings("ignore", category=RuntimeWarning)
import json, sys
import numpy as np

NOSE, LSH, RSH, LWR, RWR, LHIP, RHIP, LANK, RANK = 0, 5, 6, 9, 10, 11, 12, 15, 16


def runs_of(frames):
    out, cur = [], []
    for f in frames:
        ok = f['box'] is not None and f['kp'] is not None
        if ok and cur:
            pb, b = cur[-1]['box'], f['box']
            ok_join = abs((pb[0] + pb[2]) / 2 - (b[0] + b[2]) / 2) < 60 and abs(pb[3] - b[3]) < 40 and f['t'] - cur[-1]['t'] < .2
            if not ok_join:
                out.append(cur); cur = []
        if ok: cur.append(f)
        elif cur: out.append(cur); cur = []
    if cur: out.append(cur)
    return out


def measure(run):
    t = np.array([f['t'] for f in run])
    kp = np.array([f['kp'] for f in run], dtype=float)
    kp[kp[..., 2] < .3, :2] = np.nan
    # light temporal smoothing
    sm = kp.copy()
    for i in range(len(kp)):
        sm[i, :, :2] = np.nanmedian(kp[max(0, i - 1):i + 2, :, :2], axis=0)
    kp = sm
    H = np.nanmedian(np.nanmean(kp[:, [LANK, RANK], 1], axis=1) - kp[:, NOSE, 1])
    if not H or np.isnan(H) or H < 60:
        return None
    wr = np.nanmean(kp[:, [LWR, RWR], :2], axis=1)
    sp = np.linalg.norm(np.gradient(wr, t, axis=0), axis=1) / H
    sp[:2] = sp[-2:] = np.nan
    if np.isnan(sp).all():
        return None
    ci = int(np.nanargmax(sp))
    hip = np.nanmean(kp[:, [LHIP, RHIP], :2], axis=1)
    st = t < t[ci] - .35
    stance_front = np.nanmedian(kp[st, LANK, 1]) if st.sum() >= 3 else np.nan
    hw = (t >= t[ci] - .3) & (t <= t[ci])
    nose_rel = kp[hw, NOSE, :2]
    head = np.nanmax(np.linalg.norm(nose_rel - nose_rel[0], axis=1)) / H if len(nose_rel) > 2 else np.nan
    # backlift: highest point of the hands above the hips in the 0.6 s before contact, as a share of height
    bl = (t >= t[ci] - .6) & (t <= t[ci])
    lift = np.nanmax((hip[bl, 1] - wr[bl, 1]) / H) if bl.any() else np.nan
    stance_w = np.nanmedian(np.abs(kp[st, LANK, 0] - kp[st, RANK, 0]) / H) if st.sum() >= 3 else np.nan
    # skeleton for the site: hip-centred, height-normalised, 1.0 s before to 0.4 s after contact
    keep = np.where((t >= t[ci] - 1.0) & (t <= t[ci] + .4))[0]
    ref = hip[ci]
    sk = [[None if np.isnan(v) else round(float(v), 3) for v in ((kp[i, :, :2] - ref) / H).ravel()] for i in keep]
    return {
        't0': float(t[0]), 't1': float(t[-1]), 'contact_t': round(float(t[ci]), 3), 'height_px': round(float(H), 1),
        'peak_hand_speed': round(float(sp[ci]), 2),
        'stride': round(float((kp[ci, LANK, 1] - stance_front) / H), 3) if not np.isnan(stance_front) else None,
        'head_travel': round(float(head), 3) if not np.isnan(head) else None,
        'backlift': round(float(lift), 3) if not np.isnan(lift) else None,
        'stance_width': round(float(stance_w), 3) if not np.isnan(stance_w) else None,
        'skeleton': {'times': [round(float(t[i] - t[ci]), 3) for i in keep], 'contact_index': int(np.where(keep == ci)[0][0]), 'frames': sk},
    }


if __name__ == '__main__':
    pose = json.load(open(sys.argv[1])); out = []
    for c in pose:
        for r in runs_of(c['frames']):
            if r[-1]['t'] - r[0]['t'] < .45: continue
            m = measure(r)
            if m and m['peak_hand_speed'] > 2.0:
                m['shot'] = c['shot']; out.append(m)
    json.dump(out, open(sys.argv[2], 'w'))
    for m in out: print({k: v for k, v in m.items() if k != 'skeleton'})
