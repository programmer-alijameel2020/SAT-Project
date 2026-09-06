# SatMOT Research System

First prototype of the experiment platform for:

**Real-Time Motion-Geometry-Grounded Multi-Object Tracking for Occlusion Recovery in Satellite Video**

This implements the mathematical specification (homography-grounded motion
decomposition) end to end: robust video-only homography estimation,
platform-compensated (decomposed) target trajectories, a weighted-least-
squares residual predictor with an IRLS/Huber option, a Kalman realization
in compensated coordinates, Mahalanobis+IoU association with reactivation
gating, and occlusion-recovery / efficiency metrics -- wired into a live,
browser-based dashboard and a per-experiment file archive.

**Not executed in this session** -- delivered as source only. Everything
below is what to do on your machine.

---

## 1. Project layout

```
satmot-system/
  backend/
    app/
      geometry/         <- the actual research contribution
        homography.py         robust G_{t->t+1} estimation (RANSAC + Huber refine)
        decomposition.py      TransformChain, q_k(t) compensated coordinates
        residual_predictor.py weighted LS constant-velocity / quadratic fit, IRLS
        kalman.py              compensated-coordinate Kalman filter
        representation.py     raw / affine / homography ablation switch
      detection/
        simple_detector.py    placeholder host detector (frame-differencing)
      tracking/
        association.py        Hungarian + Mahalanobis/IoU gating
        tracker.py             MotionGroundedTracker: the full per-frame loop
      datasets/
        base.py, viso.py, sdmcar.py, demo_synth.py
      pipeline/
        orchestrator.py        runs one experiment, emits a live event per step
        events.py                canonical stage-name constants
      db/                       SQLAlchemy models + session (SQLite)
      storage/
        experiment_store.py    per-run output folder (config/results/frames/log)
      api/
        routes_experiments.py, routes_datasets.py, ws_manager.py
      main.py                    FastAPI app; also serves the frontend
    inspect_dataset.py     dataset-structure inspection tool (see Section 6)
    requirements.txt
    run.py
  frontend/
    index.html      setup screen (all parameters)
    dashboard.html    live pipeline visualization
    explorer.html      past-experiment file browser
    css/style.css
    js/{api,setup,dashboard,explorer}.js
  experiments/           <- created at runtime; one folder per run
```

## 2. Setup

```bash
cd satmot-system/backend
python3 -m venv venv
source venv/bin/activate            # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Requires Python 3.10+.

## 3. Run

```bash
python run.py
```

One instruction starts everything: FastAPI serves both the API and the
frontend from the same process/port, and your browser opens automatically
to **http://localhost:8000** once the server is confirmed ready. Use
`python run.py --no-browser` if running headless.

## 4. First run -- verify the pipeline with the synthetic demo source

On the setup screen, leave **Dataset source = demo** and click
**Create & launch experiment**. This needs no dataset path and exercises
the entire pipeline (detection -> homography -> decomposition ->
prediction -> association -> occlusion recovery) against a synthetic
sequence before you point it at real data.

## 5. Running against VISO / SDM-Car -- confirmed real layouts

These paths were confirmed against an actual extraction, not guessed:

### VISO

```
VISO/mot/mot/<category>/<seq_id>/img/000001.jpg, 000002.jpg, ...
VISO/mot/mot/<category>/<seq_id>/gt/gt.txt  (+ gt_centroid.txt)
```
where `<category>` is one of `car`, `plane`, `ship`, `train` (VISO's four
moving-object classes -- "train" here is the vehicle, not a data split).

**Set `dataset_root` to `.../VISO/mot/mot`** specifically (the actual
extracted `mot.rar`), not the VISO root folder. The scanner walks
recursively for any folder with 5+ images, so pointing it at the VISO
root would also pick up `Detection_coco_format/` and `Detection_voc_format/`
(class-organized detection-only dumps, not MOT sequences) as false
"sequences." Skip `SOT.rar` too -- it's single-object tracking data, not
relevant here.

VISO ships as `.rar`, not `.zip` -- extract with 7-Zip or WinRAR, Windows'
built-in extractor won't open it.

### SDM-Car

```
SDM-Car/train/1-1.avi          SDM-Car/train/1-1-gt.csv
SDM-Car/train/10-1.avi          SDM-Car/train/10-1-gt.csv
...
```

**Important correction from the original assumption:** SDM-Car does NOT
ship as per-frame image folders with label files -- it's paired video
(`.avi`) + CSV ground-truth files directly in each split folder (`train`/
`test`/`validation`). `SDMCarLoader` was rewritten to match this: each
sequence is one `.avi`, opened directly with OpenCV's `VideoCapture`
(same as the "raw video" path VISOLoader already supported), with ground
truth read from the matching `<seq_id>-gt.csv`.

**Set `dataset_root` to `.../SDM-Car/train`** (or `test`/`validation`).

**One open item:** the exact CSV column layout in `*-gt.csv` hasn't been
verified against a real file yet -- `SDMCarLoader.load_annotations()`
tries a header-based match first, then falls back to an assumed
`frame,id,x,y,w,h` headerless ordering if the first row is fully numeric,
and returns raw unparsed rows (rather than silently mis-parsing) if
neither pattern fits. This doesn't block running the pipeline --
`load_annotations()` isn't called anywhere in the live tracking loop, only
frame iteration is -- but paste the first few lines of one real
`*-gt.csv` and this can be locked down exactly before it's needed for
benchmark evaluation.

Once dataset_root is set: click **Scan for sequences** first before
launching a full run -- if it comes back with 0 sequences, that's telling
you the path is wrong before you waste a run on it.

## 6. If a dataset structure doesn't match

Run the included inspector against the real path in question:

```bash
cd backend
python inspect_dataset.py "/path/to/dataset/subfolder" --depth 4
```

It prints a depth-limited tree with file-type counts and a few example
filenames per folder (not every file -- sequence folders can hold
thousands of frames). Paste the output back for a loader fix.

## 7. What's a placeholder vs the real contribution

- **`geometry/*.py`** -- the actual mathematical spec, implemented
  directly. Equation numbers are cited in docstrings/comments for
  cross-checking against the source document.
- **`detection/simple_detector.py`** -- explicitly NOT the research
  contribution (the research plan states the detector is a controlled
  component). Frame-differencing blob detector so the full pipeline runs
  today; swap it for a trained detector (YOLOv8/CenterNet-style) in
  Phase 3 without touching anything downstream --
  `SimpleMotionDetector.detect()` is the only integration point.
- **Appearance re-ID (`d_app`)** -- wired into the association cost
  (`tracking/association.py`) but not populated by the placeholder
  detector. Cheap to add once a real detector/embedding exists.
- **Homography uncertainty propagation** -- set to zero for this
  prototype, exactly as the spec's own fallback note allows.

## 8. Known things to check on your machine (untested in this session)

- OpenCV/ORB API version differences across `opencv-python-headless`
  releases.
- WebSocket connections if you deploy behind a reverse proxy that doesn't
  pass through `Upgrade` headers.
- `backend/satmot_experiments.db` (SQLite) and `experiments/` are created
  automatically on first run.

## 9. GPU / device detection

`backend/app/utils/device.py` detects whether a CUDA-capable GPU is
available (via `torch`, if installed) and reports it in the dashboard's
**Compute device** stat box and in `results.json` (`compute_device` field)
for every run.

**Important: most of this system is CPU-bound by design and gains nothing
from a GPU.** Homography estimation (OpenCV ORB/RANSAC), the residual
predictor (small per-track least-squares), the Kalman filter, and the
Hungarian assignment all operate on small per-frame/per-track data --
moving that to a GPU would cost more in transfer overhead than it saves.
`opencv-python-headless` from pip also isn't built with CUDA support, so
there's no GPU path available for the homography step without compiling
OpenCV from source (not recommended given the build-tools friction
already hit installing `greenlet`).

**Where GPU actually matters:** the moment `SimpleMotionDetector` (the
CPU frame-differencing placeholder) is replaced with a real neural
detector -- YOLOv8/CenterNet-style, per Phase 3 of the roadmap -- that
component should run on GPU automatically. `detect_device()` is the
single place that decision gets made; any future detector should call it
rather than reimplementing device selection.

To enable GPU detection, install torch matching your CUDA version from
https://pytorch.org/get-started/locally/ (the correct pip command depends
on your installed CUDA toolkit/driver version, which is why it's not
pinned in `requirements.txt`). Without torch installed, the system runs
identically -- `detect_device()` just reports `backend: "none"` and
everything proceeds on CPU as it already does today.

## 10. Where this sits in the roadmap

Phase 3 ("Implementation") plus the start of Phase 4 ("Datasets,
baselines, metrics"). The geometry module is feature-complete against the
spec; what's left before real experiments is (a) verifying the SDM-Car
CSV column format against a real file, (b) swapping in a real detector,
and (c) running the raw/affine/homography ablation and the
occlusion-duration breakdown the paper's central claim depends on.
