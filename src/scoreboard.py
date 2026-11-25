"""Read the broadcast scoreboard every half second and turn it into an innings timeline."""
import cv2, re, json, sys
import numpy as np, pytesseract
from rapidocr_onnxruntime import RapidOCR

BAND = (1335, 1490, 180, 2150)      # y0, y1, x0, x1 of the scoreboard strip in a 2388x1668 frame
CELLS = {'score': (15, 85, 160, 300), 'over': (15, 85, 335, 410), 'b1': (15, 85, 440, 900),
         'b2': (15, 85, 910, 1415), 'bowl': (15, 85, 1420, 1960), 'thisover': (90, 145, 1580, 1860)}
rapid = RapidOCR()

def tess(img, psm=7, wl=None):
    g = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    cfg = f'--psm {psm}' + (f' -c tessedit_char_whitelist={wl}' if wl else '')
    return pytesseract.image_to_string(255 - g, config=cfg).strip()

def rap(img):
    r, _ = rapid(img)
    return ' '.join(x[1] for x in (r or []))

def bug_present(band):
    # the strip is saturated blue/purple; replays and transitions hide it
    hsv = cv2.cvtColor(band, cv2.COLOR_BGR2HSV)
    return float(((hsv[..., 0] > 100) & (hsv[..., 0] < 160) & (hsv[..., 1] > 120)).mean())

def read(frame):
    y0, y1, x0, x1 = BAND
    band = frame[y0:y1, x0:x1]
    blue = bug_present(band)
    if blue < 0.35:
        return {'bug': False, 'blue': round(blue, 2)}
    c = {k: band[a:b, x:z] for k, (a, b, x, z) in CELLS.items()}
    out = {'bug': True, 'blue': round(blue, 2),
           'score': rap(c['score']) or tess(c['score'], wl='0123456789-'),
           'over': rap(c['over']) or tess(c['over'], wl='0123456789.'),
           'b1': tess(c['b1']), 'b2': tess(c['b2']), 'bowl': tess(c['bowl']),
           'thisover': tess(c['thisover'], wl='0123456789WNBLD')}
    return out

def parse_bat(s):
    m = re.search(r'([A-Z][A-Z .]+?)\s+(\d{1,3})\s*\(?\s*(\d{1,3})\s*\)?\s*$', s.replace('|', ' '))
    return (m.group(1).strip(), int(m.group(2)), int(m.group(3)), '>' in s or '»' in s or '▶' in s) if m else None

if __name__ == '__main__':
    src, dst, step = sys.argv[1], sys.argv[2], float(sys.argv[3]) if len(sys.argv) > 3 else 0.5
    cap = cv2.VideoCapture(src); rows = []; t = 0.0
    dur = cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS)
    while t < dur:
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000); ok, fr = cap.read()
        if not ok: break
        r = read(fr); r['t'] = round(t, 2); rows.append(r); t += step
    json.dump(rows, open(dst, 'w'), indent=0)
    print(len(rows), 'frames read;', sum(r['bug'] for r in rows), 'with scoreboard')
