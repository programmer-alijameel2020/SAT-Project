from __future__ import annotations

import os
import json
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from ..analysis.report_generator import generate_report
from ..storage.experiment_store import EXPERIMENTS_ROOT

router = APIRouter(prefix="/api/reports", tags=["reports"])

REPORTS_ROOT = os.path.join(EXPERIMENTS_ROOT, "_reports")


class GenerateReportRequest(BaseModel):
    name_prefix: str = ""


@router.post("/generate")
def generate(req: GenerateReportRequest):
    os.makedirs(REPORTS_ROOT, exist_ok=True)
    summary = generate_report(EXPERIMENTS_ROOT, req.name_prefix, REPORTS_ROOT)
    if "error" in summary:
        raise HTTPException(400, summary["error"])
    return summary


@router.get("")
def list_reports():
    if not os.path.isdir(REPORTS_ROOT):
        return []
    out = []
    for entry in sorted(os.listdir(REPORTS_ROOT), reverse=True):
        summary_path = os.path.join(REPORTS_ROOT, entry, "summary.json")
        if os.path.isfile(summary_path):
            with open(summary_path) as f:
                out.append(json.load(f))
    return out


@router.get("/latest")
def latest_report():
    reports = list_reports()
    if not reports:
        raise HTTPException(404, "No reports generated yet.")
    return reports[0]


@router.get("/{report_id}")
def get_report(report_id: str):
    summary_path = os.path.join(REPORTS_ROOT, report_id, "summary.json")
    if not os.path.isfile(summary_path):
        raise HTTPException(404, "Report not found.")
    with open(summary_path) as f:
        return json.load(f)


@router.get("/{report_id}/file")
def get_report_file(report_id: str, path: str):
    report_dir = os.path.join(REPORTS_ROOT, report_id)
    full_path = os.path.abspath(os.path.join(report_dir, path))
    if not full_path.startswith(os.path.abspath(report_dir)):
        raise HTTPException(403, "invalid path")
    if not os.path.isfile(full_path):
        raise HTTPException(404, "file not found")
    return FileResponse(full_path)
