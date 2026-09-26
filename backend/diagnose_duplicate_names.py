"""
Diagnostic: finds experiment folders on disk whose config.json "name"
field collides with another folder's -- the report generator scans every
folder directly and does NOT deduplicate by name, so if a stale run from
an earlier session was never cleaned up before a fresh one was submitted
under the identical name, both get loaded together and every gap in that
sequence gets silently double- (or triple-, etc.) counted in every pooled
table and figure.

Usage:
    python diagnose_duplicate_names.py --name-pattern "batch_ship_045_*"
    python diagnose_duplicate_names.py   # checks EVERY experiment for duplicates
"""
import argparse
import fnmatch
import json
import os
from collections import defaultdict

EXPERIMENTS_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "experiments")
EXPERIMENTS_ROOT = os.path.abspath(EXPERIMENTS_ROOT)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name-pattern", default="*")
    args = ap.parse_args()

    by_name = defaultdict(list)
    for entry in sorted(os.listdir(EXPERIMENTS_ROOT)):
        if entry.startswith("_"):
            continue
        folder = os.path.join(EXPERIMENTS_ROOT, entry)
        config_path = os.path.join(folder, "config.json")
        results_path = os.path.join(folder, "results.json")
        if not os.path.isfile(config_path):
            continue
        with open(config_path) as f:
            config = json.load(f)
        name = config.get("name", "")
        if not fnmatch.fnmatch(name, args.name_pattern):
            continue

        n_gaps = None
        if os.path.isfile(results_path):
            with open(results_path) as f:
                results = json.load(f)
            n_gaps = (results.get("gt_anchored_evaluation") or {}).get("n_genuine_gt_gaps")

        by_name[name].append(dict(uid=entry, n_genuine_gt_gaps=n_gaps))

    n_dupes = 0
    for name, entries in sorted(by_name.items()):
        if len(entries) > 1:
            n_dupes += 1
            print(f"\nDUPLICATE: '{name}' -- {len(entries)} folders on disk:")
            for e in entries:
                print(f"    uid={e['uid']}  n_genuine_gt_gaps={e['n_genuine_gt_gaps']}")

    if n_dupes == 0:
        print(f"No duplicate names found matching '{args.name_pattern}'.")
    else:
        print(f"\n{n_dupes} duplicated name(s) found -- each one means every gap in that "
              f"run is being counted multiple times in report_generator's output. "
              f"Use reset_experiments.py on the specific STALE uid(s) (keep the one you "
              f"trust, delete the other(s) by folder inspection -- reset_experiments.py "
              f"deletes by name pattern, so for a true duplicate you'll need to remove the "
              f"stale folder directly, e.g.: Remove-Item -Recurse '<experiments_root>\\<stale_uid>')")


if __name__ == "__main__":
    main()
