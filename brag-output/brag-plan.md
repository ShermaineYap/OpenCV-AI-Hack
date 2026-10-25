# BoilGuard /brag plan

**What it is:** a range-hood camera that predicts when a pot will boil over and turns the heat down before it spills.
**For:** anyone who cooks while doing three other things.
**Sets it apart:** it *forecasts* the spill and beats the stove's heat lag, then checks it worked.
**Most impressive true claim:** 40 / 40 boil-overs prevented on the held-out (simulated) test set.
**Visual hook:** milk foam racing to the rim, then over it.
**Real UI shown:** the real OpenCV-annotated camera feed (rim ellipse, foam front, ETA panel) and the real web app's agent-decision trace and outcome tiles. All footage is the real system running on the BoilGuard simulator (seed 7).
**Tone:** polished, product-launch, dark kitchen palette from the app (#141311 bg, #ff7a4d flame, #45c19f safe, #3987e5 blue; Inter + JetBrains Mono).
**Share caption:** see share-copy.txt.

**Music:** 120 BPM (beat = 0.5 s), A minor (Am–F–C–G), 20 s. Every cut lands on a beat.

| # | Time | Scene | On screen |
|---|---|---|---|
| 1 | 0.0–3.0 | Hook | No-system camera, foam rising fast (3× speed). "Look away for ten seconds." |
| 2 | 3.0–5.0 | Spill | Foam goes over the rim. Impact on the cut. "Boil-over." |
| 3 | 5.0–7.0 | Reveal | Logo + "BoilGuard" / "sees it coming." |
| 4 | 7.0–12.0 | Same pot, side by side | Left: without (spills at 11.3 s). Right: BoilGuard annotated feed. Chips on beats: finds the rim → measures the foam → forecasts the spill → turns the heat down (synced to the real reduce_heat at 10.0 s). |
| 5 | 12.0–16.0 | The agent | "Alerts you. / Turns the heat down. / Checks it worked." beside the real web app: outcome tiles + agent decision trace. |
| 6 | 16.0–20.0 | Punchline | "40 / 40 boil-overs prevented" (held-out simulated test set) · OpenCV 5 + AWS · repo URL. |

No personal data appears. The cook name is not shown; the GitHub repo is public.
