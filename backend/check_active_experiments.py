"""
Lists every experiment currently in "running" or "pending" status on the
server -- the fastest way to see what was mid-flight when a batch
script's terminal got cut, without needing to remember or retype the
original batch command.

Usage:
    python check_active_experiments.py
"""
import sys
try:
    import requests
except ImportError:
    print("This script needs the 'requests' package: pip install requests")
    sys.exit(1)

BASE = "http://localhost:8000"


def main():
    try:
        requests.get(f"{BASE}/api/health", timeout=5).raise_for_status()
    except Exception as e:
        print(f"Could not reach the backend at {BASE} -- is 'python run.py' running? ({e})")
        sys.exit(1)

    exps = requests.get(f"{BASE}/api/experiments", timeout=15).json()
    active = [e for e in exps if e["status"] in ("running", "pending")]

    if not active:
        print("No active (running/pending) experiments -- nothing is currently mid-flight.")
        print("If you expected something to be running, it may have already finished, "
              "failed, or been interrupted cleanly before reaching the server at all.")
        return

    print(f"{len(active)} active experiment(s):\n")
    for e in active:
        print(f"  {e['name']:50s} uid={e['experiment_uid']}  status={e['status']}")

    print("\nTo check if a specific one is genuinely still progressing (vs. frozen), run:")
    print(f'  Get-Content "experiments\\<uid>\\events.jsonl" | Select-Object -Last 1')
    print("twice, ~10 seconds apart, and compare the frame number.")


if __name__ == "__main__":
    main()
