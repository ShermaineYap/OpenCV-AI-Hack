# Submission checklist (deadline: Mon 26 Oct 2026, 11:45 pm PDT = Tue 27 Oct, 2:45 pm Malaysia time)

Rules: https://opencv26.devpost.com/rules · Every entry must use OpenCV 5 for substantive image or
video analysis and run a meaningful component on AWS.

| Requirement (from the Devpost page) | Where it is | Status |
|---|---|---|
| Technical report: problem, users, architecture, OpenCV 5 implementation, AWS deployment, evaluation, limitations, responsible use | `docs/TECHNICAL_REPORT.md` (+ PDF export) | ✅ |
| Judge-accessible code repository | https://github.com/ShermaineYap/opencv-ai-hack (public) | ✅ |
| Pinned dependencies + build, deployment and test instructions | `requirements*.txt`, `Dockerfile`, `README.md`, CI workflow | ✅ |
| Architecture diagram (OpenCV 5 + AWS components, agent components) | `docs/architecture.png`, `docs/agent_workflow.png` | ✅ |
| Working web endpoint (or live screen-share) | App URL from the *Deploy to AWS* workflow | ⬜ you: deploy (step 2 below) |
| Video ≤ 5 min showing team, app working, architecture, results | script: `docs/VIDEO_SCRIPT.md` | ⬜ you: record + upload unlisted |
| Evaluation evidence incl. failure cases / limitations | `results/` + report §8-9 | ✅ |
| Agentic Vision award: workflow diagram, trace showing OpenCV output changing actions, evaluation of task success / failure handling / human control | `docs/agent_workflow.png`, trace in web app + `results/raw`, report §6 and §8 | ✅ (tick the award on Devpost) |

## Your steps (about 1-2 hours in total)

1. **Register / confirm** you joined the hackathon on Devpost and that you are in "My projects".
2. **Deploy to AWS (≈15 min).**
   1. Sign in to the AWS console (new accounts get Free Tier credits). Region **us-east-1**.
   2. IAM → Users → Create user `boilguard-deployer` → attach `AdministratorAccess` (simplest for a
      hackathon; delete the user after judging) → Security credentials → Create access key → "CLI".
   3. GitHub repo → Settings → Secrets and variables → Actions → New repository secret:
      `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY`.
   4. GitHub → Actions → **Deploy to AWS** → Run workflow (region `us-east-1`, your e-mail for alerts).
   5. When it finishes (~8-12 min), open the run → *Summary* → copy the **App URL**. Open it and run a
      scenario. Confirm the SNS e-mail subscription if you entered an e-mail.
   6. Optional: Bedrock console → Model access → make sure **Amazon Nova Lite** is available
      (if not, the app still works and uses message templates).
   7. Paste the App URL into `README.md` (top) and the Devpost form.
3. **Record the video (≈45 min)** following `docs/VIDEO_SCRIPT.md`. Keep it under 5:00. Upload to
   YouTube as *Unlisted*.
4. **Fill in Devpost** using `docs/DEVPOST.md` (copy-paste sections), attach
   `docs/architecture.png`, `docs/agent_workflow.png`, `results/fig_outcomes.png`,
   `results/fig_example_timeline.png` as gallery images, add the repo link, App URL, video link,
   and upload the technical report PDF. Opt in to the **Agentic Vision Award**.
5. Submit before the deadline. After judging, delete the stack (CloudFormation → `boilguard` → Delete)
   and the IAM user.
