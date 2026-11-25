"""Split the broadcast into camera shots using colour-histogram change between consecutive frames."""
import cv2, json, sys, numpy as np
cap = cv2.VideoCapture(sys.argv[1]); prev = None; diffs = []; ts = []; i = 0
while True:
    ok, fr = cap.read()
    if not ok: break
    t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
    small = cv2.resize(fr[160:1335], (240, 120))          # picture area only (no scoreboard, no letterbox)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    h = cv2.calcHist([hsv], [0, 1], None, [32, 16], [0, 180, 0, 256]); cv2.normalize(h, h)
    if prev is not None: diffs.append(float(cv2.compareHist(prev, h, cv2.HISTCMP_BHATTACHARYYA))); ts.append(t)
    prev = h; i += 1
d = np.array(diffs); thr = max(0.35, np.median(d) + 6 * np.median(np.abs(d - np.median(d))))
cuts = [ts[k] for k in range(len(d)) if d[k] > thr and (k == 0 or d[k] >= d[max(0, k-3):k+4].max())]
# drop cuts closer than 0.4 s (flashes, wipes)
clean = []
for c in cuts:
    if not clean or c - clean[-1] > 0.4: clean.append(c)
json.dump({'threshold': thr, 'cuts': clean, 'frames': i}, open(sys.argv[2], 'w'))
print('frames', i, 'threshold', round(thr, 3), 'shots', len(clean) + 1)
