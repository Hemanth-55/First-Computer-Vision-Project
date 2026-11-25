"""Find the striker in every live delivery shot and estimate his 17-point pose, frame by frame (YOLO11)."""
import cv2, json, sys, numpy as np
from ultralytics import YOLO

DET, POSE = YOLO('yolo11s.pt'), YOLO('yolo11m-pose.pt')
TOP, BOT = 160, 1335                     # picture area (above the scoreboard)

def striker(frame):
    """The striker in the main broadcast camera stands at the far crease: feet between y=600 and 740 in a
    2388x1668 frame, 140-270 px tall, near the middle of the pitch. The keeper behind him is shorter and higher;
    the non-striker is much lower in the picture. Returns None when no one matches (other camera angles)."""
    pic = frame[TOP:BOT]
    r = DET(pic, classes=[0], verbose=False, imgsz=1280, conf=.25)[0]
    cands = []
    for (x0, y0, x1, y1), c in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy()):
        Y1, h, cx = y1 + TOP, y1 - y0, (x0 + x1) / 2
        if 600 <= Y1 <= 740 and 140 <= h <= 270 and 1000 <= cx <= 1550:
            cands.append((h, (x0, y0 + TOP, x1, Y1, float(c), 0.)))
    return max(cands)[1] if cands else None

def pose_on(frame, box):
    x0, y0, x1, y1 = box[:4]; w, h = x1 - x0, y1 - y0
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2; S = max(w, h) * 1.5
    X0, Y0 = int(max(cx - S / 2, 0)), int(max(cy - S / 2, 0))
    crop = frame[Y0:int(cy + S / 2), X0:int(cx + S / 2)]
    k = 512 / crop.shape[0]
    big = cv2.resize(crop, None, fx=k, fy=k, interpolation=cv2.INTER_CUBIC)
    r = POSE(big, verbose=False, imgsz=512, conf=.2)[0]
    if r.keypoints is None or len(r.keypoints) == 0: return None
    # take the person whose box centre is nearest the crop centre
    bx = r.boxes.xyxy.cpu().numpy(); cc = np.array([big.shape[1] / 2, big.shape[0] / 2])
    i = int(np.argmin(np.linalg.norm((bx[:, :2] + bx[:, 2:]) / 2 - cc, axis=1)))
    kp = r.keypoints.data[i].cpu().numpy()            # 17 x (x, y, conf)
    kp[:, 0] = kp[:, 0] / k + X0; kp[:, 1] = kp[:, 1] / k + Y0
    return kp

if __name__ == '__main__':
    src, shots_file, dst = sys.argv[1:4]
    shots = [s for s in json.load(open(shots_file)) if s['delivery_view'] and s['bug'] > .9]
    cap = cv2.VideoCapture(src); out = []
    for s in shots:
        cap.set(cv2.CAP_PROP_POS_MSEC, s['start'] * 1000); fr_list = []
        while True:
            ok, fr = cap.read(); t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
            if not ok or t > s['end'] - .02: break
            b = striker(fr)
            kp = pose_on(fr, b) if b else None
            fr_list.append({'t': round(t, 3), 'box': [round(float(v), 1) for v in b[:4]] if b else None,
                            'kp': np.round(kp, 1).tolist() if kp is not None else None})
        out.append({'shot': s['shot'], 'start': s['start'], 'end': s['end'], 'frames': fr_list})
        print('shot', s['shot'], len(fr_list), 'frames,', sum(f['kp'] is not None for f in fr_list), 'with pose', flush=True)
    json.dump(out, open(dst, 'w'))
