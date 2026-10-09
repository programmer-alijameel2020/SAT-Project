"""
Reconstructs the equivalent run_batch_experiments.py command directly
from experiments already submitted under a given name prefix -- no need
to recall, search shell history, or dig up the original invocation.
Every experiment's own saved config.json already records exactly which
sequence/seed/mode/dataset it was, so the full intended campaign matrix
can be rebuilt from the server's own data.

ONE LIMITATION, stated plainly: this can only reconstruct combinations
that were ALREADY submitted at least once before the interruption. If
the original command listed sequences that the campaign never reached
yet (it runs sequentially), those won't appear here -- add them back
manually to the printed command if you know of any that are missing.

Usage:
    python reconstruct_campaign.py --name-prefix final_
"""
import argparse
import re
import sys
from collections import defaultdict

try:
    import requests
except ImportError:
    print("This script needs the 'requests' package: pip install requests")
    sys.exit(1)

BASE = "http://localhost:8000"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name-prefix", required=True,
                     help="e.g. 'final_' -- matches every experiment whose name starts with this")
    args = ap.parse_args()

    try:
        requests.get(f"{BASE}/api/health", timeout=5).raise_for_status()
    except Exception as e:
        print(f"Could not reach the backend at {BASE} -- is 'python run.py' running? ({e})")
        sys.exit(1)

    exps = requests.get(f"{BASE}/api/experiments", timeout=30).json()
    matches = [e for e in exps if e["name"].startswith(args.name_prefix)]

    if not matches:
        print(f"No experiments found with name prefix '{args.name_prefix}'. "
              f"Check the prefix (case-sensitive) and that the server has the data.")
        return

    # Group by dataset_source -- a campaign spanning both VISO and
    # SDM-Car under the same name prefix is really two separate
    # run_batch_experiments.py invocations, each needing its own command.
    #
    # Everything needed (dataset_source, sequence_name, trajectory_mode)
    # is already present directly in the single list response above --
    # no per-experiment detail fetch required for those. The one field
    # NOT in the list response is the seed, but it's embedded right in
    # the experiment's own name (run_batch_experiments.py always writes
    # "..._seed<N>_<mode>"), so a quick regex on the name already in
    # hand gets it too. This avoids what the previous version of this
    # script did: one individual HTTP request PER experiment, which for
    # a few hundred experiments made it look frozen for a minute or more
    # with zero visible progress. This version makes exactly one request
    # total, regardless of how many experiments match.
    by_dataset = defaultdict(lambda: dict(sequences=set(), seeds=set(), modes=set()))
    seed_pattern = re.compile(r"_seed(\d+)_")

    skipped_no_seed = 0
    for e in matches:
        ds = e.get("dataset_source", "unknown")
        g = by_dataset[ds]
        if e.get("sequence_name"):
            g["sequences"].add(e["sequence_name"])
        if e.get("trajectory_mode"):
            g["modes"].add(e["trajectory_mode"])
        seed_match = seed_pattern.search(e.get("name", ""))
        if seed_match:
            g["seeds"].add(int(seed_match.group(1)))
        else:
            skipped_no_seed += 1

    if skipped_no_seed:
        print(f"(note: {skipped_no_seed} matching experiment(s) had no parsable "
              f"'_seed<N>_' in their name -- their sequence/mode were still counted "
              f"above, just not their seed)\n")

    print(f"Reconstructed from {len(matches)} experiment(s) matching '{args.name_prefix}':\n")

    for ds, g in by_dataset.items():
        sequences = sorted(g["sequences"])
        seeds = sorted(g["seeds"])
        modes = sorted(g["modes"])
        print(f"=== dataset_source={ds}  ({len(sequences)} sequence(s), "
              f"{len(seeds)} seed(s), {len(modes)} mode(s)) ===")
        seq_str = " ".join(sequences)
        seed_str = " ".join(str(s) for s in seeds)
        mode_str = " ".join(modes)
        print(f"python run_batch_experiments.py --dataset-source {ds} "
              f"--name-prefix {args.name_prefix} --sequences {seq_str} "
              f"--seeds {seed_str} --modes {mode_str}")
        print()

    print("NOTE: this only includes sequences/seeds/modes that were already submitted at "
          "least once. If the original campaign intended more that it never reached before "
          "being interrupted, add them to the --sequences/--seeds/--modes lists above by hand. "
          "Also note: --max-frames isn't reconstructed (the printed commands omit it, so "
          "each sequence's full native length is used) -- add it by hand if the original "
          "campaign deliberately used a non-default cap.")
    print("\nRunning any command above will also save it for '--resume-last' from now on.")


if __name__ == "__main__":
    main()
