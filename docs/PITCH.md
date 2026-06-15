# SENTINEL — 60-second pitch + judge Q&A prep

> Demo: <https://sentinel-mule-detection-hvymcown8cyerjjes48cka.streamlit.app> · Repo: <https://github.com/vibhukrishnas/sentinel-mule-detection>

---

## The 60-second pitch (say it almost verbatim)

> **[Hook — 10s]** "Most teams on this problem will show you a 99% score. We'll show you why that number is a *lie* — and give you one you can actually defend."
>
> **[Problem + leak — 15s]** "This dataset is riddled with target leakage. A naive model reads a hidden post-event flag and scores a perfect 1.0 — impossible for real fraud. We built an automated **Data Integrity Auditor** that found it: feature F3912 plus ~580 month-stamp leaks. We removed them."
>
> **[Honest result — 10s]** "Our defensible number is **PR-AUC 0.885** — about 100× better than chance. And we proved it's honest *three* ways: a bootstrap CI of 0.84–0.95, an in-fold leak re-check at 0.91, and a leakage-sensitivity plateau."
>
> **[Live demo — 15s]** *(pick a critical account → Check score → Explain)* "Every alert gets a calibrated 0–100 score in ~35 ms, a plain-English reason, and a one-click investigation report. And when it's unsure, it **says so** and routes to a human — it auto-decides 99% at 0.1% error."
>
> **[Differentiator — 10s]** *(Escalate entire ring)* "It also surfaces the **mule ring** behind each account — one click escalates 50 look-alike accounts as a batch. Detection, explanation, containment — built honestly. That's SENTINEL."

**Timing tip:** if you only have 30s, keep Hook → Leak → "0.885, proven honest" → Ring. Drop the rest.

---

## Judge Q&A — the 6 questions they WILL ask (and your crisp answers)

**1. "Why isn't your accuracy higher / why not 0.99?"**
> Because 0.99 *is* the leakage. Our sensitivity sweep shows the genuine signal plateaus at ~0.86 — anything above that is the model reading a post-event flag. We report the defensible number, with the audit to prove what we removed. A higher number here would mean we *didn't* do our job.

**2. "Does this generalize beyond this one dataset / 81 mules?"**
> Honestly: 81 positives means real variance — which is why we report a *range* (0.81–0.89) and a bootstrap CI (0.84–0.95), not a point. We can't validate on a second dataset because the features are anonymized — no compatible schema exists. What we *can* show: the result is stable across non-overlapping folds, in-fold leak detection, and feature subsampling. Bank metadata in Phase-2 locks it down.

**3. "Is Candidate Ring #1 (50 of 81 mules) real, or an artifact of your similarity metric?"**
> Validated, not coincidental: Ring #1 is ~32× tighter than a random legit group (similarity 0.43 vs 0.01) and stable under feature subsampling (Jaccard 0.82). But we call it a **candidate** ring — true confirmation needs bank link/device/beneficiary data, which is Phase-2. We don't overclaim.

**4. "Where do the ₹ figures come from?"**
> They're illustrative assumptions — and we made them **editable**: open the cost panel and plug in the bank's real numbers; the ₹ impact recomputes live. The point isn't the exact rupee figure, it's that a missed mule costs ~46× a false alarm, so the economics favour a wide net.

**5. "How robust is it?"**
> Robust to *missing* data — the realistic failure mode — degrading gracefully (25% of features missing → still 0.66). It's *sensitive* to feature noise, which we state openly: with 81 positives the signal concentrates in a few features. That's exactly why we ship a conservative floor and the abstention layer that routes uncertain cases to a human instead of guessing.

**6. "Is this safe / what about the data?"**
> The raw bank dataset is **not** in our public repo by design — the demo runs on a small anonymized sample, and you can upload the full file to score it live without it being stored. It's decision-support for analysts, not an autonomous freeze authority. Production hardening (auth, durable audit store, rate limits) is scoped and partly built.

---

## One-liners to have ready
- **What it is:** "Leakage-proof mule detection with explainable, ring-aware containment."
- **The edge:** "We catch the leak that fakes everyone else's score."
- **The honesty flex:** "Our model knows what it doesn't know — and routes it to a human."
- **If a demo step lags:** "First load wakes the free host (~20s) — the model itself scores in 33 ms."
