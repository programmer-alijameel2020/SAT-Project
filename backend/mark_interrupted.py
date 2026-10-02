"""
Recover from a power loss / crash during a batch campaign.

Problem: an experiment that was running when the machine lost power stays
"running" in the database forever -- the process that would have marked it
completed/failed died with the machine. run_batch_experiments.py treats
"running" as "reattach and wait", so it would poll that dead run until its
one-hour timeout and never re-run it.

Fix: after power returns and BEFORE restarting the batch runner, run this
once. Every experiment still marked "running" or "pending" is necessarily an
orphan (nothing survives a reboot), so it is marked "failed" with an
explanatory message. The batch runner then re-submits exactly those runs as
fresh experiments; completed runs are untouched and skipped as before.

The orphan's partial output folder is left on disk (for auditability) but it
has no results.json, so it contributes nothing to the report.

Usage (from backend\\, with NO batch runner active):
    python mark_interrupted.py                 # dry run: list orphans
    python mark_interrupted.py --apply         # mark them failed
    python mark_interrupted.py --apply --prefix final_
"""
import argparse
import datetime

from app.db.database import SessionLocal
from app.db.models import Experiment


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="Actually update the database.")
    ap.add_argument("--prefix", default="", help="Only touch experiments whose name starts with this.")
    a = ap.parse_args()

    db = SessionLocal()
    try:
        rows = (db.query(Experiment)
                  .filter(Experiment.status.in_(["running", "pending"]))
                  .all())
        rows = [r for r in rows if str(getattr(r, "name", "") or "").startswith(a.prefix)]
        if not rows:
            print("No orphaned running/pending experiments found.")
            return
        for r in rows:
            print(f"orphan: {r.experiment_uid}  {getattr(r, 'name', '')}  status={r.status}")
        if not a.apply:
            print(f"\n{len(rows)} orphan(s). Dry run only -- re-run with --apply to mark them failed.")
            return
        now = datetime.datetime.utcnow()
        for r in rows:
            r.status = "failed"
            r.error_message = "interrupted: process died (power loss/crash) before completion"
            r.finished_at = now
        db.commit()
        print(f"\nMarked {len(rows)} orphan(s) as failed. The batch runner will re-run them.")
    finally:
        db.close()


if __name__ == "__main__":
    main()
