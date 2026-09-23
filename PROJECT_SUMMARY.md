# Project Summary — Homography-Grounded Occlusion Recovery for Satellite Video MOT

**Working title:** *Real-Time Motion-Geometry-Grounded Multi-Object Tracking for Occlusion Recovery in Satellite Video*
**Status as of this summary:** Methodology validated on one sequence; batch statistical campaign in progress; not yet publication-ready.

---

## 1. Research identity

**Core research question:** Can explicit separation of platform-induced apparent motion from residual target motion — estimated directly from video, with no ephemeris or attitude metadata — improve identity preservation through occlusion in satellite-video MOT, at real-time computational cost?

**Central claim under test:** A video-estimated homography, used to express a target's trajectory in a platform-compensated coordinate system, makes short-horizon occlusion forecasting more accurate than fitting the same forecaster to the raw pixel trajectory — with everything else in the tracker held identical, isolating the representation itself as the variable.

**The gap this targets**, established via literature pressure-testing before any code was written: existing occlusion-survival mechanisms in satellite-video tracking (Kalman filters, polynomial fitting, learned temporal models) operate on the *observed pixel-space trajectory* directly. The one paper found to partially overlap (DeTracker, Jan 2026) decouples platform motion from target motion only at the feature level, for detection-stage stabilization — not as the input representation for occlusion forecasting. SiamGM and SiamTITP, the closest analogues in spirit, are single-object trackers. No paper found combines: MOT + an explicit geometry-derived motion decomposition specifically for the forecasting step + real-time/edge efficiency reported as a headline metric alongside accuracy.

---

## 2. What was built

A complete, working research platform, not just an algorithm:

- **The geometry module** (the actual research contribution): robust video-only homography estimation (ORB + RANSAC + Huber refinement, with an affine/identity fallback chain), a `TransformChain` that maps any track's pixel history into a common platform-compensated reference frame, a weighted least-squares residual predictor (constant-velocity and quadratic, with iteratively-reweighted robust fitting), and a Kalman filter realized in the compensated coordinate system. A single switch (`raw` / `affine` / `homography`) changes only the trajectory representation the predictor consumes — everything downstream is identical, which is what makes the ablation causal.
- **A tracking-by-detection loop**: Hungarian assignment with Mahalanobis + IoU gating, occlusion-age-based reactivation, and track-confirmation logic.
- **Two dataset loaders**, built against the *actual* confirmed layouts of VISO (`category/seq_id/img/*.jpg` + `gt/gt.txt`) and SDM-Car (paired `.avi` + `.csv` files — corrected mid-project from an initial wrong assumption).
- **A full web application**: FastAPI backend, live WebSocket-driven dashboard visualizing every pipeline stage in real time, a parameter-driven experiment setup screen, a SQLite experiment database, and a per-experiment output folder (config, results, homography diagnostics, overlay frames, full event log) browsable from the UI.
- **A ground-truth-as-detector mode**, purpose-built to isolate the tracking/geometry mechanism from detector quality — a real, working placeholder frame-differencing detector exists too, explicitly marked as a stand-in for the neural detector planned in Phase 3.
- **A synthetic platform-motion injector**, precedented directly by DeTracker's own "SDM-Car-SU" approach, built after discovering that VISO's released frames carry essentially no usable platform motion.
- **A ground-truth-anchored evaluator**, computing Recovery@K exactly as originally specified in the mathematical formulation, matched against genuine ground-truth occlusion gaps rather than internal tracker bookkeeping.
- **A batch experiment campaign toolkit**: a gap-density survey across all VISO sequences, an automated multi-sequence/multi-seed/multi-mode batch runner hitting the system's own API, and an aggregator computing McNemar's exact test on paired outcomes.

---

## 3. The research journey — what was learned, in order

This is worth recording plainly because the corrections themselves are part of the scientific record, not just debugging noise.

1. **Literature pressure-test** confirmed the gap was real and not already claimed, with one close paper (DeTracker) identified and explicitly differentiated against.
2. **First working pipeline** ran end-to-end on real VISO data — but the placeholder detector over-triggered massively on real sensor noise (2,500+ detections/frame vs. a handful on synthetic test data), which surfaced two real tracker bugs: `K_max` was hardcoded and never actually controlled track death, and `min_hits_to_confirm` was accepted as a parameter but never enforced. Both were fixed.
3. **A direct measurement was added** (`platform_motion_px`) to check how much genuine platform motion VISO's released video actually contains. Result: **~0.01–0.05 px/frame mean across three different sequences** — essentially none. A direct visual diff (frame 1 vs. frame 200 of `car_001`) confirmed the background is pixel-for-pixel static. This meant the raw-vs-homography ablation had nothing to test against, regardless of mechanism quality.
4. **Literature re-examined this finding**: DeTracker's own authors built a synthetic "unstabilized" variant of their base dataset for the identical reason. This is a known issue in the field, not a flaw in this pipeline. A synthetic motion injector was built in response, deliberately modeled on that precedent.
5. **A frame-indexing bug was found and fixed**: VISO's `gt.txt` numbers frames from 1, while the frame iterator yielded 0-indexed frames — every ground-truth lookup was silently off by one frame.
6. **The internal occlusion-event count was found to be wildly inflated** — up to 330 "events" reported on a 260-frame sequence whose ground truth contains only 14 genuine occlusion gaps (confirmed independently via a standalone gap-analysis script reading `gt.txt` directly). This meant every prior ablation result was measuring detector/association noise, not real occlusion recovery.
7. **A proper ground-truth-anchored evaluator was built**, matching tracker-assigned IDs to ground-truth IDs via IoU + Hungarian assignment per frame, and computing recovery only across genuine, dataset-verified occlusion gaps.
8. **A second, subtler bug was found in that evaluator itself**: it compared warped tracker output against unwarped ground-truth positions, silently failing to match anything once synthetic motion was injected. Fixed by feeding the evaluator the same per-frame warped positions the tracker actually saw. Verified against a synthetic drift test before trusting it on real data.
9. **The first genuinely clean, valid result** was obtained on `car_001`: **homography recovered 5/14 genuine occlusion gaps (35.7%); raw recovered 3/14 (21.4%)**. Homography's recovered set was a strict superset of raw's — the two discordant cases both favored homography, none favored raw. Encouraging, but `n=14` with 2 discordant pairs is far too small to claim significance.
10. **Batch campaign infrastructure was built** to scale this properly across multiple sequences and multiple random seeds, with McNemar's exact test (the statistically correct test for this paired design) computed automatically. Currently running.

---

## 4. Current empirical result (single sequence, provisional)

| | Raw | Homography |
|---|---|---|
| Genuine GT gaps recovered | 3/14 (21.4%) | 5/14 (35.7%) |
| Long gap (21 frames, exceeds K_max) | Lost | Lost (expected — exceeds occlusion horizon) |
| Discordant cases | 0 favor raw | 2 favor homography |

A separate, genuine finding surfaced in the same data: a dense cluster of four targets around frames 90–99 showed real identity-swapping under both trajectory representations — a tracker-robustness weakness independent of the raw/homography question, worth its own investigation.

---

## 5. Honest status against publication standard

**What is methodologically sound right now:**
- Ground-truth-anchored Recovery@K evaluation (IoU + Hungarian matching to real annotated gaps)
- Paired statistical testing appropriate to the design (McNemar's, not a naive rate comparison)
- Synthetic motion injection with a direct literature precedent, not an ad hoc workaround
- A causal ablation architecture (one representation switch, everything else held fixed)

**What is not yet sufficient for a paper claim:**
1. **Every result so far uses ground-truth-as-detector** — necessary for isolating the mechanism, but not a real-world number. A real detector must replace it before any result is publishable.
2. **Synthetic motion is unvalidated against real platform motion** — the biggest open risk, not a footnote. No confirmation yet that a momentum-damped random-walk homography resembles genuine orbital/attitude-induced motion.
3. **`K_max=20` is already being exceeded by real gaps** in the one sequence tested — the hardest real cases currently can't even be evaluated.
4. **Sample size is thin** — one sequence, 14 genuine gaps, 2 discordant pairs. Not decisive in either direction yet.
5. **No external baseline** (OITrack, HP-Tracker, or even a plain non-geometric Kalman baseline on a shared protocol) — current results are an internal ablation only.
6. **Only 2 of the roadmap's ~10 planned ablation cells have been run** (raw vs. homography, one motion magnitude, one detector mode). Affine mode, motion-severity sweeps, and the decomposition-isolation ablations are all still open.
7. **No efficiency table yet** (FLOPs, parameter count, edge-hardware latency) — the paper's stated differentiating claim, currently only informal wall-clock numbers on one development machine.

**Overall assessment:** this is a rigorously *built instrument* that has already caught and corrected several of its own real methodological errors — which is exactly the right process. What it does not yet have is enough *measurements* through that instrument to support a claim. That is the current bottleneck, not the soundness of the approach.

---

## 6. In progress right now

A batch campaign (`run_batch_experiments.py` → `aggregate_results.py`) sweeping multiple VISO sequences and multiple synthetic-motion seeds per sequence, feeding a pooled McNemar's exact test across all sequences combined — the first result expected to carry real statistical weight.

## 7. Next steps, roughly in priority order

1. Let the batch campaign finish; read the pooled, statistically-tested result.
2. Decide, based on that result, whether the mechanism is worth carrying into a real-detector setting (Phase 3 of the original roadmap) or needs revisiting first.
3. Investigate the frames 90–99 dense-cluster ID-swapping as its own finding.
4. Validate (or explicitly bound as a limitation) the synthetic-motion-injection assumption against any available real unstabilized footage.
5. Run the remaining ablation cells (affine mode, motion-severity sweep, decomposition-isolation) once the headline result is established.
6. Add the efficiency table (FLOPs/params/edge latency) once a real detector is in the loop.
