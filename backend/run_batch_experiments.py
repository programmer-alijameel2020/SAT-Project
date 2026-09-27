"""
Batch experiment runner for the raw-vs-homography ablation campaign.

Submits many experiments via the already-running local API (start
`python run.py` in another terminal first), across a chosen list of
VISO sequences, trajectory modes, and synthetic-motion seeds, waiting
for each to finish before submitting the next -- kept sequential
deliberately, since this is a single-process backend and concurrent
CPU-bound runs would just contend for the same core with no real
speedup, while making failures harder to attribute.

RESUMABLE: before submitting each (sequence, seed, mode) combination,
this checks the server's own experiment list for one with the same
deterministic name. If it's already "completed", it's skipped entirely.
If it's "running" or "pending" (e.g. the orchestrator kept going on the
server after this script was Ctrl+C'd, since the server-side task is
independent of this client), this reattaches and polls the EXISTING run
instead of submitting a duplicate. Only "failed"/"cancelled"/missing
combinations get freshly submitted. This means interrupting the script
at any point and rerunning the exact same command is always safe and
picks up where it left off -- no manual bookkeeping needed.

Each (sequence, seed, mode) combination is one occlusion-recovery data
point set; running multiple seeds per sequence gives multiple
independent draws of the injected-motion realization, which is what
turns a handful of anecdotal comparisons into an actual sample large
enough for a statistical test (see aggregate_results.py).

FINAL CAMPAIGN: default modes are the six rows of representation.MODE_SPECS
and the default name prefix is "final_". The old "batch_" prefix is refused
unless --allow-legacy-prefix is given: resumability skips any COMPLETED run
with the same name, so reusing "batch_" would silently mix pre-fix results
into the final campaign.

Usage:
    python run_batch_experiments.py --sequences car_001 car_014 plane_040 --seeds 1 2 3 4 5
    python run_batch_experiments.py --sequences car_001 --seeds 42 --modes raw homography --max-frames 260

If interrupted (Ctrl+C) at any point, just run the exact same command
again -- already-completed runs are skipped automatically.
"""
import argparse
import time
import sys

try:
    import requests
except ImportError:
    print("This script needs the 'requests' package: pip install requests")
    sys.exit(1)

BASE = "http://localhost:8000"
DEFAULT_DATASET_ROOTS = {
    "VISO": r"D:\artificial_intelligence\Sat-project\datasets\VISO\mot\mot",
    "SDM-Car": r"D:\artificial_intelligence\Sat-project\datasets\SDM-Car\train",
}


def fetch_existing_by_name(prefix):
    """Returns {name: (uid, status)} for every experiment on the server
    whose name starts with `prefix` -- used to decide what to skip,
    reattach to, or freshly submit."""
    r = requests.get(f"{BASE}/api/experiments", timeout=30)
    r.raise_for_status()
    out = {}
    for exp in r.json():
        if exp["name"].startswith(prefix):
            out[exp["name"]] = (exp["experiment_uid"], exp["status"])
    return out


FINAL_MODES = ["raw_kf", "raw", "gmc", "affine", "homography", "oracle"]


def poll_until_done(uid, poll_interval=2.0, timeout=3600):
    start_time = time.time()
    while True:
        r = requests.get(f"{BASE}/api/experiments/{uid}", timeout=30)
        r.raise_for_status()
        data = r.json()
        status = data["status"]
        if status in ("completed", "failed", "cancelled"):
            return status, data
        if time.time() - start_time > timeout:
            return "timeout", data
        time.sleep(poll_interval)


def submit_fresh(name, dataset_source, dataset_root, sequence_name, mode, seed, max_frames):
    payload = dict(
        name=name,
        dataset_source=dataset_source,
        dataset_root=dataset_root,
        sequence_name=sequence_name,
        trajectory_mode=mode,
        predictor_model="constant_velocity",
        window_length=8,
        kmax_occlusion=20,
        min_hits_to_confirm=3,
        max_frames=max_frames,
        detector_mode="ground_truth",
        inject_synthetic_motion=True,
        synth_motion_px_per_frame=2.0,
        synth_motion_seed=seed,
        min_inliers=15, min_inlier_ratio=0.35, max_reproj_error=3.0,
        det_min_area=4, det_max_area=2500, det_diff_thresh=80, det_max_detections=250,
        # Saving overlay frames every 9999 frames means (with max_frames well
        # under that) only frame 0 gets saved -- skips most disk/time cost
        # across dozens of runs while still leaving one visual sanity-check
        # frame per experiment folder.
        save_frame_every=9999,
        step_delay_ms=0,
    )
    r = requests.post(f"{BASE}/api/experiments", json=payload, timeout=30)
    r.raise_for_status()
    uid = r.json()["experiment_uid"]
    r = requests.post(f"{BASE}/api/experiments/{uid}/start", timeout=30)
    r.raise_for_status()
    return uid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sequences", nargs="+", required=True,
                     help="Sequence names as reported by 'Scan for sequences', e.g. car_001 plane_040")
    ap.add_argument("--seeds", nargs="+", type=int, default=[42])
    ap.add_argument("--modes", nargs="+", default=FINAL_MODES)
    ap.add_argument("--max-frames", type=int, default=None,
                     help="Cap frames processed per run. DEFAULT (omit this flag): use each "
                          "sequence's own full length -- required for correctness on longer "
                          "sequences (e.g. car_020 has 750 frames and 382 genuine gaps; capping "
                          "at 260 silently makes every gap past frame 260 unrecoverable by "
                          "construction, deflating the absolute recovery rate for no real reason). "
                          "Pass a number only if you deliberately want a faster, truncated run.")
    ap.add_argument("--dataset-source", choices=["VISO", "SDM-Car"], default="VISO")
    ap.add_argument("--dataset-root", default=None,
                     help="Defaults to the standard path for --dataset-source if omitted.")
    ap.add_argument("--name-prefix", default="final_")
    ap.add_argument("--allow-legacy-prefix", action="store_true",
                     help="Permit the pre-fix 'batch_' prefix (NOT for the final campaign).")
    args = ap.parse_args()
    if args.name_prefix == "batch_" and not args.allow_legacy_prefix:
        print("Refusing prefix 'batch_': completed pre-fix runs under that prefix would be "
              "skipped and silently reused. Use the default 'final_' (or another new prefix).")
        sys.exit(2)
    unknown = [m for m in args.modes if m not in FINAL_MODES]
    if unknown:
        print(f"Unknown mode(s) {unknown}; valid: {FINAL_MODES}")
        sys.exit(2)
    dataset_root = args.dataset_root or DEFAULT_DATASET_ROOTS[args.dataset_source]

    try:
        requests.get(f"{BASE}/api/health", timeout=5).raise_for_status()
    except Exception as e:
        print(f"Could not reach the backend at {BASE} -- is 'python run.py' running? ({e})")
        sys.exit(1)

    existing = fetch_existing_by_name(args.name_prefix)
    n_already_done = sum(1 for _, status in existing.values() if status == "completed")
    if existing:
        print(f"Found {len(existing)} existing run(s) with prefix '{args.name_prefix}' "
              f"({n_already_done} already completed) -- these will be skipped or reattached, not re-run.\n")

    total = len(args.sequences) * len(args.seeds) * len(args.modes)
    done = 0
    results_log = []

    try:
        for seq in args.sequences:
            for seed in args.seeds:
                for mode in args.modes:
                    done += 1
                    # IMPORTANT: keep VISO's naming scheme exactly as it was
                    # (no dataset qualifier) so this doesn't break resumability
                    # for every already-completed VISO run -- only SDM-Car
                    # (a new addition, nothing completed under it yet) gets
                    # the qualifier. VISO/SDM-Car sequence names don't collide
                    # anyway ("car_001" vs "1-2"), so this is purely cosmetic
                    # for SDM-Car, not load-bearing.
                    if args.dataset_source == "VISO":
                        name = f"{args.name_prefix}{seq}_seed{seed}_{mode}"
                    else:
                        name = f"{args.name_prefix}{args.dataset_source.replace('-', '')}_{seq}_seed{seed}_{mode}"
                    existing_uid, existing_status = existing.get(name, (None, None))

                    if existing_status == "completed":
                        print(f"[{done}/{total}] {name}: already completed, skipping")
                        results_log.append(dict(name=name, sequence=seq, seed=seed, mode=mode,
                                                 uid=existing_uid, status="completed (skipped)"))
                        continue

                    if existing_status in ("running", "pending"):
                        print(f"[{done}/{total}] {name}: already {existing_status} on the server, "
                              f"reattaching instead of resubmitting ...", flush=True)
                        uid = existing_uid
                    else:
                        # not seen before, or previously failed/cancelled -> fresh submit
                        print(f"[{done}/{total}] submitting {name} ...", flush=True)
                        try:
                            uid = submit_fresh(name, args.dataset_source, dataset_root, seq, mode, seed, args.max_frames)
                        except Exception as e:
                            print(f"    -> ERROR submitting: {e}")
                            results_log.append(dict(name=name, sequence=seq, seed=seed, mode=mode,
                                                     uid=None, status="submit_error"))
                            continue

                    try:
                        status, data = poll_until_done(uid)
                    except Exception as e:
                        print(f"    -> ERROR polling: {e}")
                        results_log.append(dict(name=name, sequence=seq, seed=seed, mode=mode,
                                                 uid=uid, status="poll_error"))
                        continue

                    gt_eval = ((data.get("results_json") or {}).get("gt_anchored_evaluation")
                               if data else None)
                    n_rec = gt_eval.get("n_recovered") if gt_eval else None
                    n_gaps = gt_eval.get("n_genuine_gt_gaps") if gt_eval else None
                    print(f"    -> {status}  uid={uid}  recovered={n_rec}/{n_gaps}")
                    results_log.append(dict(name=name, sequence=seq, seed=seed, mode=mode,
                                             uid=uid, status=status,
                                             n_recovered=n_rec, n_gaps=n_gaps))
    except KeyboardInterrupt:
        print(f"\n\nInterrupted after {done}/{total}. Already-completed runs are saved on the "
              f"server -- just run this exact same command again to resume; completed and "
              f"in-progress runs will be skipped/reattached automatically, not redone.")
        sys.exit(1)

    n_failed = sum(1 for r in results_log if "completed" not in r["status"])
    print(f"\n=== BATCH COMPLETE: {total - n_failed}/{total} succeeded ===")
    if n_failed:
        print("Non-completed runs:")
        for r in results_log:
            if "completed" not in r["status"]:
                print(f"  {r['name']}: {r['status']}")
    print("\nRun aggregate_results.py next to compute the paired comparison.")


if __name__ == "__main__":
    main()