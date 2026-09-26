r"""
Standalone CLI version of the full scientific report -- generates the
exact same tables and figures (PDF + PNG) as the web app's Results page,
using the same underlying app.analysis modules, so the two can never
disagree. Useful for running headless, in a script, or just preferring
the terminal.

Usage:
    python generate_report.py --prefix batch_
    python generate_report.py --prefix "" --experiments-root "D:\...\experiments"

Output goes to <experiments_root>/_reports/<report_id>/ -- figures/ has
every plot as both .pdf and .png, tables/ has every table as both .csv
and .json, and summary.json has the headline numbers. This is the same
folder the web Results page reads from, so a report generated here shows
up there too (and vice versa).
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.analysis.report_generator import generate_report

DEFAULT_EXPERIMENTS_ROOT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "experiments"
)
DEFAULT_EXPERIMENTS_ROOT = os.path.abspath(DEFAULT_EXPERIMENTS_ROOT)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiments-root", default=DEFAULT_EXPERIMENTS_ROOT)
    ap.add_argument("--prefix", default="batch_",
                     help="Only include experiments whose name starts with this. "
                          "Pass an empty string to include every experiment on disk.")
    args = ap.parse_args()

    reports_root = os.path.join(args.experiments_root, "_reports")
    print(f"Loading experiments from {args.experiments_root} (prefix='{args.prefix}') ...")
    summary = generate_report(args.experiments_root, args.prefix, reports_root)

    if "error" in summary:
        print(f"\n{summary['error']}")
        sys.exit(1)

    report_dir = os.path.join(reports_root, summary["report_id"])
    print(f"\n=== REPORT {summary['report_id']} ===")
    print(f"Runs included:       {summary['n_runs']}")
    print(f"Genuine gaps total:  {summary['n_gaps_total']}")
    print(f"Sequences:           {summary['n_sequences']}")
    print(f"Categories:          {summary['n_categories']}")
    print()
    print("Pooled recovery by mode:")
    for mode, stats in summary.get("recovery_by_mode", {}).items():
        rate_str = f"{stats['recovery_rate']:.1%}" if stats["recovery_rate"] is not None else "n/a"
        ci_str = f"[{stats['ci_low']:.1%}, {stats['ci_high']:.1%}]" if stats["total"] else ""
        print(f"  {mode:12s}: {stats['recovered']}/{stats['total']}  ({rate_str})  {ci_str}")
    print()
    print(f"Discordant pairs:     {summary['n_discordant_pairs']} "
          f"({summary['favor_homography']} favor homography, {summary['favor_raw']} favor raw)")
    if summary["mcnemar_p_value"] is not None:
        sig = ("***" if summary["mcnemar_p_value"] < 0.001 else
               "**" if summary["mcnemar_p_value"] < 0.01 else
               "*" if summary["mcnemar_p_value"] < 0.05 else "n.s.")
        # Very small p-values (routine at this sample size) round to a
        # useless 0.000000 under fixed-decimal formatting -- show
        # scientific notation for anything below 1e-4, where that
        # starts to matter, and report the robust log10 form alongside
        # it (directly citable in a paper: "p < 10^-88").
        p = summary["mcnemar_p_value"]
        p_str = f"{p:.6f}" if p >= 1e-4 else f"{p:.4e}"
        log10_str = (f"  (log10 p = {summary['mcnemar_log10_p_value']})"
                     if summary.get("mcnemar_log10_p_value") is not None else "")
        print(f"McNemar exact p-value: {p_str}  ({sig}){log10_str}")
    if summary["odds_ratio"] is not None:
        print(f"Odds ratio:            {summary['odds_ratio']:.3f}  "
              f"[{summary['odds_ratio_ci_low']:.3f}, {summary['odds_ratio_ci_high']:.3f}]")

    print(f"\nFigures ({len(summary['figures'])}, each as .pdf + .png):")
    for fig in summary["figures"]:
        status = f"ERROR: {fig['error']}" if "error" in fig else "OK"
        print(f"  {fig['name']:32s} {status}")
    print(f"\nTables ({len(summary['tables'])}, each as .csv + .json):")
    for tbl in summary["tables"]:
        print(f"  {tbl['name']}")

    print(f"\nAll files written to: {report_dir}")


if __name__ == "__main__":
    main()
