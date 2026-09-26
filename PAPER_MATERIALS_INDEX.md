# Paper Materials Index

**Final validated report ID: `20260925_122258_6d7a2e`**
(380 runs, 32,580 genuine gaps, 37 sequences, 5 categories — duplicate-folder bug fixed, p-value rounding bug fixed. Use this one, not any earlier report ID mentioned in this chat's history.)

All paths below are relative to:
```
D:\artificial_intelligence\Sat-project\experiments\_reports\20260925_122258_6d7a2e\
```

---

## Tier 1A — Report outputs

**1. Five (now eight) summary tables** — `tables\`:
- `summary_by_mode.csv`
- `summary_by_mode_bucket.csv`
- `summary_by_sequence.csv`
- `summary_by_category.csv`
- `latency_summary.csv`
- `paired_gap_records.csv` *(new)*
- `platform_motion_by_sequence.csv` *(new)*
- `internal_vs_gt_occlusion_counts.csv` *(new)*

Each also has a `.json` twin with identical content, same folder.

**2. Eleven figures, vector PDF** — `figures\*.pdf` (PNGs sit alongside for quick viewing, but the PDFs are what go in the paper):
`overall_recovery`, `recovery_by_bucket`, `recovery_by_sequence`, `forest_odds_ratio`, `discordant_breakdown`, `recovery_vs_duration`, `recovery_by_category`, `sequence_bucket_heatmap`, `latency_distribution`, `motion_injection_validation`, `homography_fallback_rates`.

**3. Raw per-gap reproducibility export** — `tables\paired_gap_records.csv`. One row per genuine gap, both conditions' outcomes side by side. Exact p-value: **1.3657×10⁻⁸⁸** (log₁₀p ≈ −87.865) — pulled directly from `summary.json` in this same folder, not rounded.

---

## Tier 1B — Configuration (values, not files — pulled directly from source)

| Parameter | Value |
|---|---|
| m₀ / r₀ / ε₀ | 15 / 0.35 / 3.0 px |
| τ_H (RANSAC threshold) | 3.0 px |
| κ (conditioning gate) | **not implemented in code** — describe only the 3-condition gate actually shipped |
| Huber δ (homography) | 2.0 |
| Huber δ (residual IRLS) | 5.0 — separate value, don't conflate |
| IRLS iterations | fixed 3, not convergence-based |
| Window L / K_max | 8 / 20 |
| Kalman q_a / R | 1.0 / 4.0·I₂ |
| Association gate α | χ²(0.95, df=2) ≈ 5.99, widened ×(1+0.05·min(age,30)) |
| Cost weights | λ_g=1.0, λ_IoU=3.0, λ_app=0.0 (never populated) |
| **min_hits_to_confirm** | **effectively 1** for every reported run (forced override whenever `detector_mode="ground_truth"`, which is all of them) |
| Motion injection | 2.0 px/frame nominal, momentum=0.85, pure translation, **no sweep performed** |
| Detector mode | `ground_truth` for every reported result |

Source files, if you want to quote line numbers: `app/geometry/homography.py`, `app/tracking/association.py`, `app/geometry/kalman.py`, `app/geometry/residual_predictor.py`, `app/pipeline/orchestrator.py` (search `effective_min_hits`).

---

## Tier 1C — Audit evidence

**9.** Per-sequence measured vs. configured motion: `tables\platform_motion_by_sequence.csv`.
Visual proof pair (re-export directly, not saved as standalone files anywhere):
`datasets\VISO\mot\mot\car\001\img\000001.jpg` and `...\000200.jpg`.

**10.** Annotation-format sample lines (already verified, quote directly — or re-pull fresh from):
`datasets\VISO\mot\mot\car\001\gt\gt.txt` (comma, `[x,y,w,h]`)
`datasets\VISO\mot\mot\ship\045\gt\gt.txt` (whitespace, `[x1,y1,x2,y2]`)

**11.** `tables\internal_vs_gt_occlusion_counts.csv` — per-sequence inflation ratio, direct evidence for the 20–60× claim.

**12.** Genuine gap counts including zero-gap sequences — **not** inside the report folder, these live in `backend\`:
`backend\gt_gap_survey_VISO.csv`
`backend\gt_gap_survey_SDMCar.csv`

---

## Tier 1D — History (text, ready to paste into the paper's methodology notes)

> The partial-run pooled recovery rate reported early in development (19.0%/21.9%) was measured before a frame-truncation bug was fixed: `car_020`, which alone constituted ~91% of that early sample, had only its first 260 of 750 frames evaluated, so most of its 382 genuine occlusion gaps fell outside the evaluated window and were counted as unrecoverable by construction, not by tracking failure. After correcting the evaluation window to each sequence's full native length and re-running the affected sequence, the pooled recovery rate rose to the reported 49.3%/56.0%. A second, independent issue — duplicate experiment folders sharing an identical name across separate campaign sessions for one sequence (`ship_045`) — was found and corrected via an automated duplicate-detection audit before finalizing results; the paired statistical tests (which key on exact gap identity) were unaffected by this issue, but simple pooled counts were, and have been corrected.

---

## Tier 1E — Hardware (you need to supply this — not something I can generate)

Run and record:
```powershell
Get-ComputerInfo | Select-Object CsProcessors, OsName, OsVersion, CsTotalPhysicalMemory
pip show opencv-python-headless numpy scipy fastapi
python --version
```

---

## Tier 2 / Tier 3 — not run yet

`affine` mode: infrastructure ready, not executed (185 additional runs, optional, discussed separately). MOTA/IDF1/HOTA, injection-magnitude sweep, parameter ablations, real-detector integration, qualitative frame overlays: all genuinely new work, none pulled from existing data.
