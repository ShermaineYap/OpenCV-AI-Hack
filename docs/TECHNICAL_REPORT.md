# BoilGuard: physics-informed boil-over prediction and prevention with OpenCV 5 and AWS

**OpenCV AI Competition 2026 (powered by AWS) — technical report**
Author: Shermaine Yap Shi Min, Asia Pacific University of Technology and Innovation (APU), Malaysia
Code: https://github.com/ShermaineYap/opencv-ai-hack · Paths: Overall + Agentic Vision Award

---

## 1. Problem

Cooking is the leading cause of home fires. In the US, NFPA attributes nearly half (49%) of home
fires to cooking equipment, about 158,400 per year in 2017-2021, with 470 deaths, 4,150 injuries
and $1.15 billion of direct damage; **unattended cooking is by far the leading factor** [1]. In
Malaysia, the Fire and Rescue Department (Bomba) summarises the cause of most home fires as
"3L": *lalai, lupa, leka* — negligence, forgetting, distraction [2].

A boil-over is the everyday, high-frequency version of the same failure. Milk, pasta water,
rice, curry and porridge foam over within seconds of the boil; the spill puts out gas flames
(leaving gas flowing), bakes onto ceramic and induction glass, trips induction hobs, and
scalds. Existing smart-stove features are **reactive** (detect a spill or smoke afterwards) or
**timer-based**. Nothing watches the pot and predicts *when* it will spill.

The timing matters because stoves have **thermal inertia**: after the heat is turned down,
foam keeps rising for about 1-2 s on induction, 4-6 s on ceramic and 5-8 s on electric coils.
A warning has to come before that lag runs out, or it is useless.

**Users.** People who cook while multitasking (parents, students, shift workers), older adults
with memory issues and their carers, and appliance makers: camera-equipped hoods already exist
(e.g. GE's Kitchen Hub streams a downward view of the cooktop [4]), so the hardware is in kitchens.

**Prior art.** Patents describe detecting that water *is boiling* [5] or monitoring boiling for
cooking assistance [6], and hoods that react to smoke or heat. BoilGuard's difference is
(a) predicting *when* a foaming pot will spill, with calibrated uncertainty, early enough to beat
the hob's heat lag, and (b) closing the loop: act, verify from video, escalate. **Benefit:** fewer spills, burns and gas-flame
outages; less cleaning; peace of mind to step away briefly.

## 2. What BoilGuard does

A camera in the range hood looks down at the pot. BoilGuard

1. **finds the pot rim** and measures how far the liquid/foam surface is below it, in millimetres;
2. **predicts** the probability that the pot will spill *within the hob's heat lag* and the
   expected time to overflow, using a physics-informed foam-growth model;
3. **acts**: alerts the cook, lowers the heat early enough to beat the lag (with a cancel window,
   approval or notify-only mode), then **verifies from the video** that foam growth reversed and
   escalates to burner OFF if not; when steam hides the rim it turns the **hood fan** on to see better;
4. **explains**: Amazon Bedrock writes the phone notification and an incident report from the
   numbers.

## 3. Architecture

![architecture](architecture.png)

The same Python package and container image run at the stove (edge) and on AWS. On AWS, an
**AWS Lambda** container (FastAPI via the Lambda Web Adapter, behind a Function URL) runs the
full pipeline for the web demo and for uploaded clips; **AWS IoT Core** carries device commands
over MQTT (`boilguard/<device>/cmd`) and an IoT topic rule archives every actuation to **Amazon S3**
for audit; **Amazon SNS** delivers alerts; **Amazon Bedrock** (Nova Lite, Converse API) writes
messages; **Amazon CloudWatch** receives metrics via Embedded Metric Format, a dashboard and an
error alarm; everything is one **AWS SAM** template deployed by a GitHub Actions workflow.

Design choice: the safety loop does not depend on the cloud. Vision, prediction and the policy are
deterministic and run in ~20 ms per analysed frame on one CPU core, so they can run on the hood's
own processor; the cloud adds remote actuation, notifications, audit, narration and observability.

## 4. OpenCV 5 implementation

All perception is classical OpenCV 5 (`opencv-python-headless==5.0.0.93`); no training data is
needed, which matters for a safety product that must work on any pot on day one.

| Step | OpenCV 5 functions | Why |
|---|---|---|
| Rim detection (once) | median of 15 frames, `createCLAHE`, `GaussianBlur`, `Canny`, `dilate`, `findContours`, `fitEllipse`, `HoughCircles` fallback | Candidate ellipses scored by edge support (360 samples), size and roundness; the largest well-supported one is the pot's outer edge. |
| Spill-line refinement | `remap` (elliptical polar), `Sobel`, per-angle sub-pixel peaks, `fitEllipse` + trimming | Hood lights sit next to the camera, so the up-facing rim lip is a thin bright ridge. Its inner edge is where liquid spills. |
| Stabilisation | `phaseCorrelate` + `createHanningWindow` on a rim-only annulus | Cancels hood vibration; hands or spoons crossing the rim give weak/large peaks and are rejected. |
| Unwrap | `remap` with precomputed elliptical-polar maps | A rising surface becomes a front moving outward along each row. |
| Steam | per-angle atmospheric scattering model I = J·t + A(1−t) fitted on the lip + stove band; `medianBlur`, `GaussianBlur` | Steam is haze. Inverting the dehazing model restores the wall's appearance; 1 − t is a steam index. |
| Foam front | `subtract`, `multiply`, `transform`, `sqrt` (colour distance, ±1 px tolerant), vectorised change point via cumulative sums | The inner wall is static background: per angle, find where "differs from the calm wall" becomes "matches it". |
| Robust estimate | median and MAD over 360 angles; lip-coverage test | Occluders cover a sector, foam covers all angles; overflow = rim lip covered *and* front at the rim. |
| Depth | pinhole model: r_surface / r_rim = H / (H + d) | Converts the front's radius to millimetres below the rim using the camera height. |
| Boiling onset | `calcOpticalFlowFarneback` on the liquid region | Agitation flags boiling before foam forms. |
| Overlay | `ellipse`, `circle`, `putText`, H.264 via ffmpeg | Annotated video for the cook and the judges. |

## 5. Prediction: physics-informed Kalman filter

Protein- and starch-stabilised foam grows **exponentially** at onset (bubble films trap gas that
nucleates more bubbles) and then **saturates** or collapses as drainage catches up; with heater
power P_e, dh/dt = P_e g h (1 − h/h_max) − λ (1 − P_e) h. In log space this is a growth rate
k(t) that drifts, so BoilGuard tracks x = [log(foam + h0), k, dk/dt] with `cv2.KalmanFilter`,
rebuilding the transition matrix for every variable time step. Measurement noise comes from the
vision confidence and steam index, with a 4-σ innovation gate.

The forecast is probabilistic: 400 samples from the posterior are propagated with the
kinematic model (growth acceleration is not extrapolated upward — a physical prior) and each
crossing time of the rim is solved in closed form. This gives P(spill within lag + 3 s),
P(spill within 20 s) and ETA percentiles. A plateau predicts "no overflow", which is what keeps
heat cuts away from simmering pots.

## 6. Agent (perception → decision → action → verification)

![agent workflow](agent_workflow.png)

The decision policy is deterministic and auditable; every decision is logged with the perception
and forecast that caused it (see the trace in the web app and `result.json`). Visual evidence
changes the next action in five ways: (1) boiling onset raises the analysis rate from 2 Hz to
8-10 Hz (active perception and 3-4× less compute while idle); (2) steam or low confidence turns
the hood fan on and makes the policy act earlier; (3) a rising P(spill within 20 s) alerts the
cook; (4) a high P(spill within the heat lag) lowers the heat, with the human-control mode deciding
how; (5) after acting, the agent checks from video that the growth rate turned negative, and
escalates to burner OFF + alarm if it did not. Operating thresholds (P ≥ 0.7, foam ≥ 40% of the
calm freeboard; two consecutive frames when confidence < 0.4) were chosen on a separate dev set
by offline replay (`eval/tune.py`), never on the test set.

Amazon Bedrock (Nova Lite via the Converse API) writes the notification and the end-of-session
report from structured numbers only; it is never in the safety path, has a 4 s timeout and a
template fallback.

## 7. AWS deployment

«AWS_SECTION»

## 8. Evaluation

«EVAL_SECTION»

## 9. Limitations

«LIMITATIONS_SECTION»

## 10. Responsible use

* **Privacy by design.** Only the pot region is analysed; an optional privacy mask blurs
  everything outside 1.45× the rim in saved video. Frames are not stored by default on the edge;
  the cloud demo stores annotated runs in a private, encrypted S3 bucket with a 7-day lifecycle
  and serves them via 1-hour presigned URLs. Bedrock prompts contain numbers, never images.
* **The cook stays in charge.** Three autonomy modes; the default announces the action with a
  3 s cancel window; if the cook takes over, the agent stands down. BoilGuard only ever
  *reduces* heat — restoring it is always a human decision.
* **Fail safe, not fail silent.** Low visibility is detected and reported; cloud errors never
  block the local loop; device commands are rate limited and audited in S3; IAM is least-privilege
  (IoT publish restricted to `boilguard/*`).
* **Honest claims.** All quantitative results are on simulated scenes and are labelled as such.
  Automatic heat control must be validated on real stoves, with appliance-maker safety
  certification, before anyone relies on it. BoilGuard is an assistant, not a replacement for
  attention, smoke alarms or a fire extinguisher.

## 11. Reproducibility

«REPRO_SECTION»

## References

1. NFPA, *Home Cooking Fires* (2017-2021 averages), as reported by WKGC, 4 Oct 2023:
   https://www.wkgc.org/2023/10/04/cooking-fires-are-the-leading-cause-of-home-fires/
2. Utusan Malaysia, "3L punca kebakaran kediaman sendiri", 11 Apr 2023 (Bomba Sungai Besi):
   https://www.utusan.com.my/gaya/2023/04/3l-punca-kebakaran-kediaman-sendiri/
3. AWS Lambda function URL auth (Oct 2025 permission change):
   https://docs.aws.amazon.com/lambda/latest/dg/urls-auth.html
4. GE Kitchen Hub (over-the-range smart hub with a cooktop-facing camera), e.g.
   https://androidcentral.com/save-680-ge-kitchen-hub-early-black-friday-deal-best-buy
5. US Patent 9,330,469, "Systems and methods for boil detection":
   https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/9330469
6. US Patent 12,137,837, "Systems and methods for real-time monitoring of boiling fluid for food
   processing assistance": https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/12137837
