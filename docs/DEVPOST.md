# Devpost submission text (copy-paste)

**Project name:** BoilGuard
**Tagline (≤ 200 chars):** A range-hood camera that predicts when a pot will boil over and turns the heat down before it spills — OpenCV 5 vision, a physics-informed forecaster and an agent on AWS.

**Built with:** opencv, python, numpy, fastapi, aws-lambda, aws-iot-core, amazon-bedrock, amazon-nova, amazon-s3, amazon-sns, amazon-cloudwatch, aws-sam, aws-cloudformation, amazon-ecr, docker, github-actions, kalman-filter

**Links:** App URL «APP_URL» · Code https://github.com/ShermaineYap/opencv-ai-hack · Video «VIDEO_URL»

**Special awards:** opt in to **Agentic Vision Award**.

---

## Inspiration

In Malaysia, firefighters sum up most home fires in three words — *lalai, lupa, leka* (negligence,
forgetting, distraction). In the US, cooking causes nearly half of all home fires and unattended
cooking is by far the leading factor. The everyday version is the boil-over: you look away for ten
seconds and the milk, pasta water or curry is over the rim, the gas flame is out and the hob is
burnt. Turning the knob at that moment is too late, because the stove keeps heating for several
seconds after you turn it down. I wanted a system that sees the spill *coming* — early enough to beat
the stove's own heat lag.

## What it does

A camera above the stove (range hoods with cameras already exist) watches the pot. BoilGuard:

* finds the pot rim and measures, in millimetres, how far the foam is below it;
* forecasts the **probability of spilling within the hob's heat lag** and the time to overflow
  using a physics-informed model of foam growth;
* **acts**: alerts the cook, lowers the heat (with a 3-second cancel window, "ask me first" or
  "notify only" modes), **verifies from the video** that the foam is going down, and switches the
  burner off with an alarm if it is not;
* when steam hides the pot it turns the **hood fan on to see better** and acts more cautiously;
* sends commands through **AWS IoT Core** (audited in S3), alerts through **Amazon SNS**, and uses
  **Amazon Bedrock (Nova)** to write a plain-English notification and incident report.

The live web app lets anyone run a boil (milk, pasta water, curry, soup; induction, ceramic, coil;
light to heavy steam; cook nearby or away) and see the annotated camera view, the forecast and every
agent decision. You can also upload your own top-down clip.

## How I built it

**Perception — OpenCV 5, no training data needed.** Rim: CLAHE, Canny, `findContours` and
`fitEllipse`, scored by edge support, then refined to the inner edge of the bright rim lip with an
elliptical-polar `remap` and a per-angle sub-pixel ellipse fit. Every frame: rim-anchored
`phaseCorrelate` cancels hood vibration; an atmospheric scattering (dehazing) model fitted on the
rim lip removes steam haze; because the pot's inner wall is static background, a vectorised
per-angle change-point search finds the foam front; a median over 360 angles rejects hands and
spoons; pinhole geometry turns the front's radius into millimetres; Farneback optical flow flags
boiling onset. ~20 ms per frame on one CPU core.

**Prediction — physics, not a black box.** Foam grows exponentially and then saturates. A
`cv2.KalmanFilter` tracks log-foam, its growth rate and the change of that rate, with measurement
noise from the vision confidence. Monte-Carlo samples of the posterior give P(spill within heat lag)
and ETA; a plateau predicts "no spill", which keeps the heat on for a simmer.

**Agent.** A deterministic perceive → decide → act → verify → escalate loop with three human-control
modes and adaptive 2-10 Hz analysis. Every decision is logged with the evidence that caused it.

**AWS.** One SAM template: Lambda container image (FastAPI through the Lambda Web Adapter) behind a
Function URL, IoT Core device topic + topic rule archiving every command to S3, SNS, Bedrock Converse,
S3 with encryption and a 7-day lifecycle, CloudWatch Embedded Metric Format metrics, a dashboard and
an error alarm. GitHub Actions runs the tests, deploys and smoke-tests the live endpoint.

**Evaluation.** With no real-stove dataset, I built a physically grounded boil-over simulator
(logistic foam growth, heater thermal lag, steam, hands, tilt, camera shake, JPEG noise) that is
also a closed-loop environment. Thresholds were tuned on a dev set and reported on a disjoint test
set against baselines (no system, rim-threshold alarm, linear extrapolation).

«RESULTS_SHORT»

## Challenges I ran into

* **Steam.** It fooled the first version into seeing an overflow. Treating steam as haze and fitting
  the dehazing model on the always-visible rim lip fixed it and gave a steam index for free.
* **Hands over the pot** dragged the image registration; registering only on the rim annulus, with
  sanity limits, and robust statistics over angles solved it.
* **Telling a simmer from a spill early.** Early on, exponential and saturating growth look the
  same. A probabilistic forecast plus a fill gate, tuned on a separate dev set, gave a good trade-off.
* **Heat lag.** Acting at "foam near the rim" is too late on electric hobs; the decision has to use
  the hob's lag, which is why the forecast is "spill within lag + 3 s".

## Accomplishments that I'm proud of

«ACCOMPLISHMENTS»

## What I learned

Classical computer vision is still the right tool when you can describe the physics: modelling the
wall as background, steam as haze and the camera as a pinhole gave millimetre measurements without
any training data. And an honest evaluation (dev/test split, baselines, failure cases) changed the
design more than any single algorithm.

## What's next

A real-stove study with a hood camera on induction, ceramic and gas hobs; a Raspberry Pi / Graviton
edge build; integration with Matter smart plugs and IoT-enabled hobs; more liquids (rice, porridge,
jam); and a caregiver mode that notifies family members.
