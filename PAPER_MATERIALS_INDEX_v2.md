# Paper Materials Index (v2 — includes affine mode)

**Final validated report ID: `20260926_193823_5111f2`**

All paths below are relative to:
```
D:\artificial_intelligence\Sat-project\experiments\_reports\20260926_193823_5111f2\
```

**Scope of this final report:** 565 runs (380 raw+homography, 185 affine), 37 sequences, 5 categories, 3 trajectory modes.

---

## What changed since the last index (v1)

- Added the full `affine` mode across all 37 sequences (185 additional runs) — the "single most important addition" the paper-writing chat itself flagged.
- **Key new finding:** affine and homography are statistically indistinguishable pooled (56.1% vs. 56.0%), both significantly beat raw (49.3%). The `ship_045` reversal holds for **both** compensation modes equally — ruling out "it's specifically homography's flexibility" as the explanation.
- Fixed a real crash: `paired_gap_records.csv`'s column list was hardcoded to two modes and broke the moment affine data existed. Now dynamic.
- Fixed a real visualization gap: five of the eleven figures (`overall_recovery`, `recovery_by_bucket`, `recovery_by_sequence`, `recovery_vs_duration`, `recovery_by_category`) were silently still only plotting `raw`/`homography` even after affine data existed, because the mode lists inside the plotting functions were hardcoded. Now dynamic across however many modes are present. (`forest_odds_ratio` and `discordant_breakdown` remain intentionally raw-vs-homography only — extending McNemar/odds-ratio machinery to 3-way needs a different statistical design, not a quick patch; flagged as Tier 2/future work if you want three-way pairwise comparisons.)

---

## Tier 1A — Report outputs

**Tables** (`tables\`, each `.csv` + `.json`): `summary_by_mode`, `summary_by_mode_bucket`, `summary_by_sequence`, `summary_by_category`, `latency_summary`, `paired_gap_records`, `platform_motion_by_sequence`, `internal_vs_gt_occlusion_counts`.

**Figures** (`figures\*.pdf`, vector): `overall_recovery`, `recovery_by_bucket`, `recovery_by_sequence`, `forest_odds_ratio`, `discordant_breakdown`, `recovery_vs_duration`, `recovery_by_category`, `sequence_bucket_heatmap`, `latency_distribution`, `motion_injection_validation`, `homography_fallback_rates`.

**Exact p-value:** 1.3657×10⁻⁸⁸ (log₁₀p ≈ −87.865) — from `summary.json`, not rounded.

---

## Tier 1B — Configuration (unchanged from v1, still accurate)

| Parameter | Value |
|---|---|
| m₀ / r₀ / ε₀ | 15 / 0.35 / 3.0 px |
| τ_H | 3.0 px |
| κ (conditioning gate) | **not implemented** in shipped code |
| Huber δ (homography / IRLS) | 2.0 / 5.0 — separate values |
| IRLS iterations | fixed 3 |
| Window L / K_max | 8 / 20 |
| Kalman q_a / R | 1.0 / 4.0·I₂ |
| Gate α | χ²(0.95, df=2) ≈ 5.99, widened ×(1+0.05·min(age,30)) |
| Cost weights | λ_g=1.0, λ_IoU=3.0, λ_app=0.0 (never populated) |
| min_hits_to_confirm | **effectively 1** for every reported run |
| Motion injection | 2.0 px/frame nominal, momentum=0.85, translation only, no sweep |
| Detector mode | `ground_truth` throughout |

---

## Tier 1C — Audit evidence

- Platform motion per sequence: `tables\platform_motion_by_sequence.csv`
- Visual proof pair: `datasets\VISO\mot\mot\car\001\img\000001.jpg` / `...\000200.jpg`
- Annotation format samples: `car_001`: `1,1,582,24,6,9,1,-1,-1,-1` (comma, xywh); `ship_045`: `1 1 752 125 774 136 -1 -1 -1 -1` (whitespace, corners)
- Inflation evidence: `tables\internal_vs_gt_occlusion_counts.csv`
- Zero-gap sequences included: `backend\gt_gap_survey_VISO.csv`, `backend\gt_gap_survey_SDMCar.csv`

## Tier 1D — History (paste into methodology notes)

> Two measurement-artifact corrections occurred during validation, both caught by automated audits before finalizing results, neither affecting the core paired statistical tests. First, an early partial-run recovery estimate (19%) was measured before a frame-evaluation-window truncation bug was fixed; correcting it raised the estimate to the reported 49%. Second, duplicate experiment folders sharing an identical name were found for one sequence across separate campaign sessions; the paired McNemar/odds-ratio tests (which key on exact gap identity) were unaffected, but simple pooled counts were corrected accordingly.

## Tier 1E — Hardware (still needs your input)

```powershell
Get-ComputerInfo | Select-Object CsProcessors, OsName, OsVersion, CsTotalPhysicalMemory
pip show opencv-python-headless numpy scipy fastapi
python --version
```

---

## What files to actually supply to the new paper-writing chat

The new chat starts with **zero access** to your local machine or this project's files — it only sees what you paste or upload directly into it. Supply, in this order:

1. **`MASTER_PROMPT_PAPER_WRITING_v2.md`** (paste as the opening message, in full) — this alone gives the new chat everything needed to start the venue conversation and begin drafting: the research framing, what was built, and the final numbers already summarized in prose and tables.

2. **This index file** (`PAPER_MATERIALS_INDEX_v2.md`) — as a follow-up message or attachment, so the new chat knows exactly which local files exist and where, in case it asks you to go fetch something specific (e.g., "paste the contents of `summary_by_category.csv`").

3. **Upload directly, as actual files, once the new chat asks for them** (don't front-load all of these blindly — let the chat request what it actually needs as it drafts each section):
   - `figures\*.pdf` — the actual figures for embedding in the paper.
   - `summary_by_category.csv` and `summary_by_sequence.csv` — small enough to paste inline if asked.
   - The two frame images (`000001.jpg`/`000200.jpg`) if it wants the visual stabilization proof for a figure.
   - **Do not upload `paired_gap_records.csv` wholesale** — it's the full per-gap record (tens of thousands of rows); if the new chat wants to spot-check it, paste a `head -20` sample instead.

4. **Do not bother uploading** the actual codebase/zip unless the new chat specifically asks for an equation or implementation detail beyond what's in Tier 1B's config table above — the paper-writing chat's job is to write the paper from the results and the described method, not to re-derive it from source.
