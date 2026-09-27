"""
Inspect a set of runs sharing a name prefix (e.g. the 6-mode smoke test).

Answers two questions from the runs' own files:
  1. Did the NEW code run?   provenance in config.json, estimator_model in
     results.json, per-frame model/H in homography_log.json
  2. Do the modes actually behave differently?  gap-level outcomes, tracker IDs
     at each gap, ID switches, and whether the logged transforms differ.

Usage:  python inspect_smoke.py ..\\experiments --prefix smoke_
"""
import argparse
import json
from pathlib import Path

import numpy as np


def load(p: Path):
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("experiments_root")
    ap.add_argument("--prefix", default="smoke_")
    a = ap.parse_args()

    runs = {}
    for cfg_path in Path(a.experiments_root).rglob("config.json"):
        cfg = load(cfg_path)
        if not cfg or not str(cfg.get("name", "")).startswith(a.prefix):
            continue
        d = cfg_path.parent
        runs[cfg.get("trajectory_mode")] = dict(
            dir=d, cfg=cfg, res=load(d / "results.json"), hlog=load(d / "homography_log.json"))

    if not runs:
        raise SystemExit(f"No runs with prefix {a.prefix!r} under {a.experiments_root}")

    order = [m for m in ["raw_kf", "raw", "gmc", "affine", "homography", "oracle"] if m in runs]
    print("=== 1. Did the new code run? ===")
    for m in order:
        r = runs[m]
        prov = r["cfg"].get("provenance") or {}
        res = r["res"] or {}
        h0 = (r["hlog"] or [{}])[min(1, len(r["hlog"] or [{}]) - 1)]
        print(f"{m:11s} commit={str(prov.get('git_commit'))[:10]} dirty={prov.get('git_dirty')} "
              f"estimator_model={res.get('estimator_model')} "
              f"log_has_model={'model' in h0} log_has_H={'H' in h0} "
              f"frame1_mode={h0.get('mode')} fallback={res.get('homography_fallback_rate')}")

    print("\n=== 2. Do the modes differ? ===")
    def gaps(m):
        ev = ((runs[m]["res"] or {}).get("gt_anchored_evaluation") or {})
        return {(g["gt_id"], g["start"], g["end"]): g for g in ev.get("gap_details", [])}, ev

    ref = order[0]
    g_ref, _ = gaps(ref)
    for m in order:
        g, ev = gaps(m)
        keys = sorted(set(g) & set(g_ref))
        disc = sum(g[k]["recovered"] != g_ref[k]["recovered"] for k in keys)
        id_diff = sum((g[k]["tracker_id_before"], g[k]["tracker_id_after"]) !=
                      (g_ref[k]["tracker_id_before"], g_ref[k]["tracker_id_after"]) for k in keys)
        long_lost = sum(1 for k in keys if g[k]["duration"] >= 20 and not g[k]["recovered"])
        print(f"{m:11s} recovered={ev.get('n_recovered')}/{ev.get('n_genuine_gt_gaps')} "
              f"id_switches={ev.get('general_id_switches')} "
              f"outcome_diff_vs_{ref}={disc} tracker_id_diff_vs_{ref}={id_diff} "
              f"lost_with_duration>=20={long_lost}")

    print("\n=== 3. Logged transforms, mean |H_a - H_b| over frames ===")
    def Hs(m):
        return np.array([h["H"] for h in (runs[m]["hlog"] or []) if "H" in h], dtype=float)
    for a_, b_ in [("affine", "homography"), ("gmc", "affine"), ("homography", "oracle"), ("raw", "homography")]:
        if a_ in runs and b_ in runs:
            A, B = Hs(a_), Hs(b_)
            if len(A) and len(A) == len(B):
                print(f"{a_} vs {b_}: mean abs diff = {np.abs(A - B).mean():.6f}, "
                      f"identical logs = {np.array_equal(A, B)}")
            else:
                print(f"{a_} vs {b_}: transforms not comparable (lengths {len(A)} / {len(B)})")


if __name__ == "__main__":
    main()
