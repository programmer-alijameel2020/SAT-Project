Paste this as the opening message of a new chat to set full context for paper writing.

---

# Master Prompt: Paper Writing for Satellite Video MOT Occlusion-Recovery Research

## Who I am / working style

I'm Ali Jameel Hashim (also publishing as "Al-Mousawi"), Chief System Programmer and Software Architect at the Communications and Media Commission (CMC) of Iraq, PhD in Computer Engineering (University of Tabriz). Ten peer-reviewed publications including IEEE TNNLS and IEEE TCBB papers.

Working preferences:
- LaTeX for all academic writing; I compile documents myself.
- Prefer artifacts/files over inline code or text for anything substantial.
- I insist on honest, rigorous reporting — including results that underperform a baseline, mixed effects, or dataset limitations, if that's what's true. Do not manufacture positive results or soften genuine null/negative findings.
- Applications, hybridizations, and principled modifications to existing techniques are legitimate contributions if executed rigorously — the bar is "clearly differentiated, honestly validated, technically sound," not "nothing like this has ever existed."
- When uncertain about current literature or venue metrics, search rather than guess.
- On writing register: top-tier journals want an hourglass structure — Abstract/Introduction accessible broadly, Method fully precise with no hand-holding (one intuitive sentence before each formal block, then full rigor), Experiments technical but scannable, Discussion/Conclusion pulling back to plain implications again.

## The research, in brief

**Locked title:** *Real-Time Motion-Geometry-Grounded Multi-Object Tracking for Occlusion Recovery in Satellite Video*

**Core research question:** Can explicit separation of platform-induced apparent motion from residual target motion — estimated directly from video, no ephemeris/attitude metadata required — improve identity preservation through occlusion in satellite-video MOT, at real-time computational cost?

**The gap** (established via literature pressure-testing before implementation): existing occlusion-survival mechanisms in satellite-video MOT fit the *observed pixel-space trajectory* directly. The nearest paper (DeTracker, Jan 2026) decouples platform motion from target motion only at the feature level, for detection-stage stabilization, not forecasting. SiamGM/SiamTITP are single-object trackers, not MOT. **This literature scan is now months old relative to when you're reading this — re-verify nothing has been published in the interim before finalizing Related Work.**

## What was built

A full research platform: robust video-only homography estimation (ORB+RANSAC+Huber, affine/identity fallback), a platform-compensated coordinate system, a weighted-LS residual predictor + Kalman filter, a three-way trajectory-representation switch (`raw`/`affine`/`homography`), a ground-truth-as-detector evaluation mode (isolates the tracking mechanism from detector quality), a synthetic platform-motion injector (precedented by DeTracker's own "SDM-Car-SU"), a ground-truth-anchored Recovery@K evaluator (IoU+Hungarian matching to genuine annotated occlusion gaps, not internal tracker bookkeeping), and a resumable batch campaign with full statistics: Wilson CIs, exact McNemar's test (log-space computation, citable to any magnitude — e.g. p≈10⁻⁸⁸), paired odds ratios with CI, forest plots, category/sequence/duration-bucket breakdowns.

**Two dataset-level findings worth their own sentence in the paper**, discovered during validation:
- VISO's own ground-truth annotation format is **not consistent across categories** — `car_*` sequences use comma-delimited `[x,y,w,h]`; `ship_045` uses whitespace-delimited `[x1,y1,x2,y2]` corners. A format-auto-detecting parser handles both (verified numerically and by direct visual inspection of overlaid boxes on real ship wakes).
- VISO's released video is essentially platform-motion-free (~0.01–0.05 px/frame measured, and pixel-identical background 199 frames apart) — a known field-wide issue (see DeTracker's SDM-Car-SU), addressed here by injecting a controlled, validated synthetic drift (momentum-damped 2D random walk, translation only, no rotation/zoom modeled) rather than relying on the native footage.

## Final validated empirical results

**Report ID: `20260926_193823_5111f2`**

**Scope:** 37 sequences across 5 categories — VISO (13 `car`, 1 `ship`, 4 `plane`, 1 `train`) + SDM-Car (18 sequences, all vehicles, a second independent satellite/pipeline — Luojia 3-01 vs. VISO's Jilin-1). ~16,290 genuine ground-truth occlusion gaps per mode, 3 trajectory-representation modes × 5 seeds each.

**Pooled recovery by mode** (ground-truth-as-detector, synthetic motion injection at 2.0 px/frame):

| Mode | Recovery | 95% CI |
|---|---|---|
| affine | 56.1% | [55.3%, 56.8%] |
| homography | 56.0% | [55.2%, 56.8%] |
| raw | 49.3% | [48.5%, 50.0%] |

Raw-vs-homography McNemar exact test: **p ≈ 1.37×10⁻⁸⁸** (log₁₀p ≈ −87.87), odds ratio 2.106 [1.953, 2.271], n=3,078 discordant pairs (2,087 favor homography, 991 favor raw).

**Key finding #1 — the core effect, cross-platform:** Platform-motion compensation (either affine or full homography) significantly and substantially improves occlusion recovery for vehicle targets over raw pixel-space trajectory fitting. This replicates independently across VISO's 12 car sequences AND SDM-Car's 18 sequences on a different satellite/pipeline — most individually significant on their own, all pointing the same direction on the forest plot.

**Key finding #2 — affine captures essentially all the benefit:** Full projective homography (56.0%) and simple affine compensation (56.1%) are statistically indistinguishable from each other, both clearing raw by ~7pp. This reframes the contribution: the value is in compensating for platform motion at all, not specifically in the fidelity of the geometric model — and affine is cheaper to estimate, a genuine efficiency angle.

**Key finding #3 — a genuine, now-broadened boundary condition:** `ship_045` shows a significant *reversal* (raw beats both affine and homography, ~86% vs. ~23% for both compensation modes, non-overlapping CIs) — and critically, **affine reverses just as much as homography does**, ruling out "it's specifically homography's extra flexibility" as the explanation. The honest, current hypothesis: geometric compensation of any kind may hurt recovery for large, high-contrast, more steadily-moving targets (ships) versus small, erratically-moving ones (vehicles) — but this rests on **one sequence only** (n=22 discordant pairs, replicated across seeds, not across independent ship sequences — VISO has no second usable ship sequence to test generalization against). State this precisely as a scoped, well-replicated single-sequence finding, not a category-level claim.

**Mechanistic detail (duration-stratified heatmap):** the vehicle advantage is not uniform across occlusion duration — it concentrates most strongly in the **medium-duration bucket (6–15 frames)**, consistent with short occlusions being trivially bridged regardless of representation (ceiling effect) and very long occlusions degrading both representations' short-horizon linear extrapolation similarly.

**Known limitations to state honestly:**
- All results use ground-truth-as-detector (isolates the mechanism deliberately; no real-world deployed-detector claim yet — that's explicit future work).
- No injection-magnitude sweep performed (single fixed 2.0 px/frame nominal value throughout).
- No MOTA/IDF1/HOTA computed yet, no comparison against external published baselines (OITrack, HP-Tracker, etc.) — this is currently a rigorous internal ablation, not yet benchmarked against the field.

**Validation/audit trail worth a methods-section footnote:** two real bugs were found and corrected during this campaign — (1) a frame-truncation artifact that deflated an early partial-run recovery estimate (19%→49% after fixing), and (2) duplicate experiment folders under an identical name for one sequence, caught by an automated audit and corrected before finalizing (the paired statistical tests were unaffected by this, since they key on exact gap identity; only simple pooled counts needed correction). This is evidence of rigor, not something to hide.

## What I need from you in this chat

**First: recommend a target venue**, using the *actual* results above, not the framing from months ago. Weigh general ML venues (IEEE TNNLS, Neural Networks, Applied Soft Computing) against remote-sensing-specific venues (IEEE TGRS, IEEE JSTARS, ISPRS Journal of Photogrammetry and Remote Sensing) — search for current impact factors/scope/acceptance patterns rather than assume. Given the category-dependent finding and the two dataset-level discoveries (annotation-format heterogeneity, platform-motion stabilization), a remote-sensing audience may value this paper's specific contributions more than a general ML audience would see them as niche — but pressure-test this, don't just take my word for it.

**Then: help me write the paper**, section by section, in LaTeX, once venue is settled: Introduction → Related Work (explicit differentiation vs. DeTracker/SiamGM, re-verified current) → Method (full rigor — note: the original math spec included a κ conditioning-number gate term that was never actually implemented; describe only what's actually in the shipped code) → Experiments → Ablations (raw/affine/homography triplet is the headline ablation) → Discussion/Limitations (ship_045 finding, ground-truth-detector caveat, single-sequence scoping) → Conclusion.

Start with the venue recommendation before touching a line of the paper.
