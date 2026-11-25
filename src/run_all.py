"""Run the whole pipeline on a broadcast video.

    python src/run_all.py path/to/innings.mov

Each step writes to outputs/work/ and is skipped if its output already exists (delete a file to re-run that step).
Pose tracking is the slow step: about 20 minutes on a laptop CPU, a few minutes on a Colab GPU.
"""
import subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
W = ROOT / 'outputs' / 'work'; W.mkdir(parents=True, exist_ok=True)
video = sys.argv[1]
S = lambda n: str(ROOT / 'src' / n)
steps = [
    ('scoreboard_raw.json', [S('scoreboard.py'), video, W / 'scoreboard_raw.json', '0.5']),
    ('timeline.json',       [S('timeline.py'), W / 'scoreboard_raw.json', W / 'timeline.json']),
    ('scenes.json',         [S('scenes.py'), video, W / 'scenes.json']),
    ('shots.json',          [S('shots.py'), video, W / 'scenes.json', W / 'scoreboard_raw.json', W / 'shots.json']),
    ('pose_raw.json',       [S('pose.py'), video, W / 'shots.json', W / 'pose_raw.json']),
    ('strokes_raw.json',    [S('strokes.py'), W / 'pose_raw.json', W / 'strokes_raw.json']),
]
for out, cmd in steps:
    if (W / out).exists():
        print('skip  ', out); continue
    print('run   ', out, flush=True)
    subprocess.run([sys.executable, *map(str, cmd)], check=True)
subprocess.run([sys.executable, S('build_outputs.py'), W, ROOT / 'outputs'], check=True)
subprocess.run([sys.executable, S('charts.py')], check=True)
print('done: outputs/metrics.json and figures/')
