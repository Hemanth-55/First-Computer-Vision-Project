"""Label each camera shot: is it the main delivery camera (batter at the far end, 4+ small players)?"""
import cv2, json, sys
import numpy as np
from ultralytics import YOLO

if __name__ == '__main__':
    video, scenes_file, board_file, dst = sys.argv[1:5]
    det = YOLO('yolo11s.pt')
    sc = json.load(open(scenes_file)); board = json.load(open(board_file))
    cap = cv2.VideoCapture(video)
    end = cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS)
    b = [0] + sc['cuts'] + [end]; out = []
    for k in range(len(b) - 1):
        a, z = b[k], b[k + 1]; hs, cnt = [], []
        for t in (np.linspace(a + .15, z - .15, 3) if z - a > .4 else [(a + z) / 2]):
            cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000); ok, fr = cap.read()
            r = det(fr[160:1335], classes=[0], verbose=False, imgsz=1280, conf=.3)[0]
            h = (r.boxes.xyxy[:, 3] - r.boxes.xyxy[:, 1]).cpu().numpy(); cnt.append(len(h)); hs += list(h)
        bug = [x['bug'] for x in board if a <= x['t'] < z]
        s = {'shot': k, 'start': round(a, 2), 'end': round(z, 2), 'persons': float(np.median(cnt)),
             'med_h': float(np.median(hs)) if hs else 0., 'bug': float(np.mean(bug)) if bug else 0.}
        s['delivery_view'] = bool(s['persons'] >= 4 and s['med_h'] < 420)   # wide main camera: several small players
        out.append(s)
    json.dump(out, open(dst, 'w'), indent=1)
    print(len(out), 'shots;', sum(s['delivery_view'] for s in out), 'delivery views')
