"""
Safely delete experiments by exact name or name pattern -- removes BOTH
the database row (via the API's DELETE endpoint) AND the on-disk
experiment folder, keeping the two in sync.

This matters because the two failure modes of doing it by hand are both
bad in different ways:
  - Deleting only the on-disk folder (e.g. by hand in File Explorer)
    leaves the database row behind. run_batch_experiments.py's
    resumability check reads the database, sees "completed", and skips
    resubmitting it -- so the experiment just silently vanishes from
    every future report with no error.
  - Deleting only the database row (e.g. via the Past Results page's
    Delete button) leaves the OLD folder on disk. A fresh run with the
    same name then creates a SECOND folder alongside it. The report
    generator scans every folder on disk by name prefix, not by database
    ID, so both the old (possibly stale/truncated) and new data get
    loaded together -- silently double-counting that sequence's gaps in
    every pooled table and figure.

This script does both deletions together, atomically per experiment, so
neither failure mode can happen.

Usage:
    python reset_experiments.py --name-pattern "batch_car_020_*" --yes
    python reset_experiments.py --name-pattern "batch_car_020_*"   # dry run (no --yes): lists what WOULD be deleted, deletes nothing
"""
import argparse
import fnmatch
import os
import shutil
import sys

try:
    import requests
except ImportError:
    print("This script needs the 'requests' package: pip install requests")
    sys.exit(1)

BASE = "http://localhost:8000"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name-pattern", required=True,
                     help="Glob-style pattern matched against experiment names, "
                          "e.g. 'batch_car_020_*' or 'batch_ship_045_seed1_*'")
    ap.add_argument("--yes", action="store_true",
                     help="Actually delete. Without this flag, only lists what would be deleted.")
    args = ap.parse_args()

    try:
        requests.get(f"{BASE}/api/health", timeout=5).raise_for_status()
    except Exception as e:
        print(f"Could not reach the backend at {BASE} -- is 'python run.py' running? ({e})")
        sys.exit(1)

    r = requests.get(f"{BASE}/api/experiments", timeout=30)
    r.raise_for_status()
    all_experiments = r.json()

    matches = [e for e in all_experiments if fnmatch.fnmatch(e["name"], args.name_pattern)]

    if not matches:
        print(f"No experiments match pattern '{args.name_pattern}'. Nothing to do.")
        return

    print(f"{'Would delete' if not args.yes else 'Deleting'} {len(matches)} experiment(s) "
          f"matching '{args.name_pattern}':\n")
    for e in matches:
        print(f"  {e['name']:45s} uid={e['experiment_uid']}  status={e['status']}")

    if not args.yes:
        print("\nDry run only -- nothing deleted. Re-run with --yes to actually delete "
              "both the database record and the on-disk folder for each.")
        return

    # Get output_dir for each before deleting the DB row (the delete
    # endpoint only returns a message, not the row's data).
    detail_by_uid = {}
    for e in matches:
        try:
            dr = requests.get(f"{BASE}/api/experiments/{e['experiment_uid']}", timeout=15)
            dr.raise_for_status()
            detail_by_uid[e["experiment_uid"]] = dr.json()
        except Exception as ex:
            print(f"  WARNING: could not fetch details for {e['experiment_uid']}: {ex}")

    print()
    n_db_deleted, n_folder_deleted, n_errors = 0, 0, 0
    for e in matches:
        uid = e["experiment_uid"]
        detail = detail_by_uid.get(uid)
        output_dir = detail.get("output_dir") if detail else None

        try:
            requests.delete(f"{BASE}/api/experiments/{uid}", timeout=15).raise_for_status()
            n_db_deleted += 1
        except Exception as ex:
            print(f"  ERROR deleting database row for {uid}: {ex}")
            n_errors += 1
            continue

        if output_dir and os.path.isdir(output_dir):
            try:
                shutil.rmtree(output_dir)
                n_folder_deleted += 1
            except OSError as ex:
                print(f"  ERROR deleting folder {output_dir}: {ex}")
                n_errors += 1
        else:
            print(f"  NOTE: no on-disk folder found for {uid} (already gone, or never wrote results)")

    print(f"\nDone: {n_db_deleted} database record(s) deleted, {n_folder_deleted} folder(s) deleted"
          + (f", {n_errors} error(s)" if n_errors else ""))
    print("\nThese experiments are now fully gone -- rerunning run_batch_experiments.py with "
          "the same name(s) will submit them fresh rather than skipping or duplicating them.")


if __name__ == "__main__":
    main()
