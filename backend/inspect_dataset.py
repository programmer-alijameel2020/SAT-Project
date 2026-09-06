"""
Dataset structure inspector.

Prints a depth-limited directory tree. For any folder that contains files,
it summarizes them by extension (count) and shows a few example filenames,
instead of dumping every file -- which matters here since VISO/SDM-Car
sequence folders can hold thousands of frames.

Usage:
    python inspect_dataset.py "D:\\artificial_intelligence\\datasets\\VISO\\mot"
    python inspect_dataset.py "D:\\artificial_intelligence\\datasets\\SDM-Car\\train" --depth 5
"""
import argparse
import os
from collections import Counter


def walk(path: str, depth: int, max_depth: int, sample_files: int):
    if depth > max_depth:
        return
    indent = "  " * depth
    try:
        entries = list(os.scandir(path))
    except PermissionError:
        print(f"{indent}[permission denied]")
        return

    dirs = sorted([e for e in entries if e.is_dir()], key=lambda e: e.name)
    files = sorted([e for e in entries if e.is_file()], key=lambda e: e.name)

    for d in dirs:
        print(f"{indent}[DIR] {d.name}")
        walk(d.path, depth + 1, max_depth, sample_files)

    if files:
        ext_counts = Counter(os.path.splitext(f.name)[1].lower() or "(no ext)" for f in files)
        for ext, count in sorted(ext_counts.items(), key=lambda x: -x[1]):
            print(f"{indent}  ({count} files  *{ext})")
        for f in files[:sample_files]:
            print(f"{indent}  e.g. {f.name}")
        if len(files) > sample_files:
            print(f"{indent}  ... and {len(files) - sample_files} more")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", help="Root directory to inspect")
    ap.add_argument("--depth", type=int, default=4, help="Max recursion depth (default 4)")
    ap.add_argument("--sample", type=int, default=4, help="Example filenames to show per folder (default 4)")
    args = ap.parse_args()

    root = os.path.abspath(args.path)
    if not os.path.isdir(root):
        print(f"ERROR: '{root}' is not a directory.")
        return
    print(f"== {root} ==")
    walk(root, 0, args.depth, args.sample)


if __name__ == "__main__":
    main()
