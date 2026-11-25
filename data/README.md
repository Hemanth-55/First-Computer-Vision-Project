# Data

The input is the broadcast highlights of Hemanth Kumar Gopi's innings for Thiruvallur in the S.S. Rajan Trophy
(TNCA inter-district T20), published by FanCode under the title *"73 off 44! Hemanth Kumar G guides Thiruvallur to 177"*.

The footage belongs to the broadcaster and is **not stored in this repository**. Everything in `outputs/` is data the
pipeline derived from it: scoreboard readings, camera-shot boundaries and body keypoints.

To reproduce the results, save the highlights as `innings.mov` (2388×1668 screen recording, about 146 s) and run
`python src/run_all.py innings.mov`. The scoreboard position and main-camera geometry are set for this broadcast's layout.
