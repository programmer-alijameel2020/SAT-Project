"""
Local filesystem browsing for the dataset-path picker.

This backend runs locally on the same machine as the browser using it, so
it can safely expose a read-only directory listing to let the setup
screen offer a folder-picker instead of requiring the user to retype an
absolute path every time. This is intentionally NOT exposed for arbitrary
remote deployments -- if this backend is ever deployed somewhere the
frontend user shouldn't have filesystem visibility into, this endpoint
should be disabled or authenticated.
"""
from __future__ import annotations

import os
import string
from fastapi import APIRouter, HTTPException

router = APIRouter(prefix="/api/fs", tags=["filesystem"])


@router.get("/list")
def list_directory(path: str = ""):
    """
    path="" -> on Windows, returns available drive letters as entries;
               on POSIX, returns '/'.
    path="C:\\" or "/some/dir" -> returns immediate subdirectories.
    """
    if not path:
        if os.name == "nt":
            drives = [f"{d}:\\" for d in string.ascii_uppercase if os.path.exists(f"{d}:\\")]
            return dict(path="", parent=None, entries=[
                dict(name=d, path=d, type="drive") for d in drives
            ])
        else:
            path = "/"

    if not os.path.isdir(path):
        raise HTTPException(404, f"'{path}' is not a directory")

    try:
        entries = []
        for name in sorted(os.listdir(path)):
            full = os.path.join(path, name)
            if os.path.isdir(full):
                # Surface a quick hint if this folder looks like a dataset
                # sequence (helps the user recognize when to stop
                # descending) without fully scanning it.
                try:
                    peek = os.listdir(full)[:50]
                except (PermissionError, OSError):
                    peek = []
                looks_like_images = any(
                    p.lower().endswith((".jpg", ".jpeg", ".png")) for p in peek
                )
                looks_like_video = any(
                    p.lower().endswith((".avi", ".mp4", ".mov", ".mkv")) for p in peek
                )
                entries.append(dict(
                    name=name, path=full, type="dir",
                    hint="images" if looks_like_images else ("video" if looks_like_video else None),
                ))
    except PermissionError:
        raise HTTPException(403, f"Permission denied reading '{path}'")

    parent = os.path.dirname(path.rstrip("\\/")) if path not in ("/",) else None
    if os.name == "nt" and len(path.rstrip("\\/")) <= 3:
        parent = ""  # go back to drive list from a drive root

    return dict(path=path, parent=parent, entries=entries)
