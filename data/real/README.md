# Real-footage check (optional)

The quantitative evaluation in this repo is on the physically simulated benchmark
(`boilguard/sim.py`), because exact ground truth for "when did it spill" is
available there. To check BoilGuard on real footage:

1. Record (or download, with a licence that allows it) a top-down video of a pot:
   whole rim in view, camera 30-60 cm above the rim, recording starts before boiling.
2. Put it here, e.g. `data/real/milk_01.mp4`, and add a line to `labels.csv`:

   ```
   file,cam_height_mm,heater,overflow_s
   milk_01.mp4,450,ceramic,83.4
   ```
   `overflow_s` is the second at which foam first crosses the rim (leave empty if it
   never does). Read it off a video player frame by frame.
3. Run

   ```
   python -m boilguard.cli analyze data/real/milk_01.mp4 --cam-height 450 --heater ceramic
   ```
   which writes an annotated video and a JSON trace next to the input, or use the
   "Analyse your clip" tab of the web app.

Safety when filming: stay at the stove, keep the heat knob in reach, and use a pot you
are happy to clean. BoilGuard's automatic heat control must be validated on real
stoves before anyone relies on it.
