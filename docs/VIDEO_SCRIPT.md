# 5-minute video script (record with screen capture + your voice)

The rules ask the video to show **the team, the application working, its architecture,
and its principal results**, in at most 5 minutes. Record your screen with OBS, Loom,
Windows Game Bar (Win+G) or QuickTime, and your webcam for the first 15 seconds.
Upload to YouTube as **Unlisted** and paste the link into Devpost.

Have these open before you start: the deployed web app (App URL), this repo on GitHub,
`docs/architecture.png`, `docs/agent_workflow.png`, `results/fig_outcomes.png`,
`results/fig_example_timeline.png`, and the AWS console (IoT Core > MQTT test client
subscribed to `boilguard/+/cmd`, and the CloudWatch dashboard `boilguard`).

---

**0:00-0:15 · You (webcam)**
"Hi, I'm Shermaine Yap, a computing student at Asia Pacific University in Malaysia.
This is BoilGuard: a camera above your stove that sees a pot about to boil over and
turns the heat down before it happens."

**0:15-0:45 · The problem (one slide or the app header)**
"Unattended cooking is the leading cause of home kitchen fires, and boil-overs are the
everyday version: milk or pasta water foams over in seconds, puts out gas flames,
burns onto glass hobs and scalds. Smart-stove features today react when something is
already wrong. BoilGuard predicts *when* the spill will happen, early enough to beat
the stove's own heat lag, which is up to 8 seconds on an electric hob."

**0:45-2:15 · Live demo (web app, Run a scenario)**
1. Pick *Milk*, *Ceramic*, *Moderate steam*, *Will boil over: Yes*, *Cook: Away*,
   mode *Act with cancel window*, tick *Mirror commands to AWS IoT Core*. Press
   **Turn on the stove**.
2. While it runs (~30 s): "Each request runs the full pipeline in AWS Lambda: OpenCV 5
   vision, a Kalman forecaster and the agent, on a physically simulated boil."
3. Play the annotated video. Point at: the cyan rim ellipse, the coloured foam front,
   the side panel (freeboard, ETA, probability), the action ticker.
4. Show the charts: "measured surface distance (blue) tracks the truth (dashed); the
   chance of spilling within the heat lag rises; the agent turned the heat to LOW
   here, then *verified* that foam growth reversed."
5. Switch to the AWS IoT MQTT test client: "the `set_heat` command arrived on
   `boilguard/demo-stove/cmd`; a topic rule archives every actuation to S3 for audit."
6. Show the incident report card: "written by Amazon Bedrock (Nova) from the numbers
   only; the decision itself is deterministic."
7. Run the same seed with *Will boil over: No*: "on a simmer that plateaus, the
   forecast says no overflow and it leaves the heat alone."

**2:15-3:15 · How it works (architecture.png, then agent_workflow.png)**
"Perception is classical OpenCV 5, so it runs on a CPU at the stove: rim detection
with Canny, contours and fitEllipse; an elliptical polar unwrap with remap; phase
correlation to cancel hood vibration; a haze model from dehazing, fitted on the rim
lip, to see through steam; and a per-angle change-point search, because the inner
wall of the pot is static background. Pinhole geometry turns the foam front into
millimetres below the rim. Prediction uses physics: foam grows exponentially then
saturates, tracked with cv2.KalmanFilter, and Monte-Carlo sampling gives the
probability of spilling within the hob's heat lag. The agent alerts, lowers the heat,
verifies from video, escalates if needed, and when steam hides the rim it switches the
hood fan on to see better. The cook stays in control with three autonomy modes."

**3:15-4:30 · Results (fig_outcomes.png, fig_example_timeline.png, fig_forecast_error.png)**
"I evaluated on 60 held-out simulated scenes, tuned only on a separate dev set. With
no system, all 40 overflow-prone pots spill. A rim-threshold alarm with a cook nearby
still lets 88% spill, and even BoilGuard's own warnings alone let 80% spill, because
the stove's heat lag plus a six-second human reaction is simply too slow. That's why the
agent acts: in auto mode BoilGuard prevented all 40 boil-overs, with the cook nearby or
away, and cut the heat unnecessarily on one of fourteen safe pots. It gives a quiet
heads-up about eight seconds before the spill, measures the surface to about 3 mm, and
runs in under 20 ms per frame on one CPU core.
Honest failure cases: on dark tomato soup the rim was mistaken for the liquid edge in
3 of 60 scenes, and soup accuracy is about half as good as milk. All numbers are from
simulation; the next step is a real-stove study."

**4:30-5:00 · Responsible use + close**
"Only the pot region is analysed, a privacy mask can blur everything else, prompts to
Bedrock contain numbers not images, every command is audited, and heat is only ever
reduced: turning it back up is always the cook's decision. BoilGuard: see it coming,
turn it down. Thank you."
