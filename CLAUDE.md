# First Computer Vision Project: session context

Rebuilds Hemi's 73 off 44 innings (S.S. Rajan Trophy, Thiruvallur) from 146 s of broadcast highlights: scoreboard OCR, shot detection, YOLO11 player detection and 17-point pose.
Claude: read this file, then `notes/`, at the start of every session.

## The project
- Run: `pip install -r requirements.txt` (plus the tesseract binary), then `python src/run_all.py innings.mov`. The video isn't in the repo (see `data/README.md`). Each step caches to `outputs/work/`, so charts rebuild without the video.
- Pipeline: `scoreboard.py` → `scenes.py` → `shots.py` → `pose.py` → `strokes.py` → `timeline.py` → `build_outputs.py` / `charts.py`. Notebook: `notebooks/first_computer_vision_project.ipynb`.
- Headline: 3,976 frames, 246 scoreboard readings, 9 strokes tracked; stance varies 7.6% of height vs 18.2% at contact.
- Footage is a FanCode broadcast: never commit or redistribute it. YOLO11 is AGPL-3.0.
- Downstream: case study + innings-vision lab on the portfolio, and `scripts/data/cv-poses.json` in the profile repo.

## Rules
- Apple-level quality: minimal, professional, clear to a layman at a glance. Speak to tech people, no gimmicks.
- Never describe projects as done at BU.
- Never commit secrets, raw licensed data or large media.

## Session routine
- Start: `/session-start` · End: `/session-end` (log, commit, `git pull --rebase`, push)
- Several sessions can share this repo; the notes are the shared memory. Facts come only from the repo, the notes and Hemi. Ask, don't guess.

## Related
- Hub: https://github.com/Hemanth-55/Tamil- (index of all repos in `notes/repos.md`)
- Portfolio: https://github.com/Hemanth-55/Portfolio-Website (live: hemanthkumargopi.vercel.app)
