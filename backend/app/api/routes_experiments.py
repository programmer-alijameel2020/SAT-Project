from __future__ import annotations

import os
import uuid
import asyncio
import datetime
from fastapi import APIRouter, Depends, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import Optional, List

from ..db.database import get_db, SessionLocal
from ..db.models import Experiment
from ..storage.experiment_store import ExperimentStore, EXPERIMENTS_ROOT
from ..pipeline.orchestrator import run_experiment, request_cancel
from .ws_manager import manager

router = APIRouter(prefix="/api/experiments", tags=["experiments"])


class ExperimentCreateRequest(BaseModel):
    name: str
    dataset_source: str          # "VISO" | "SDM-Car" | "demo"
    dataset_root: Optional[str] = ""
    sequence_name: Optional[str] = None
    trajectory_mode: str = "homography"    # raw | affine | homography
    predictor_model: str = "constant_velocity"
    window_length: int = 8
    kmax_occlusion: int = 20
    min_hits_to_confirm: int = 3
    max_frames: Optional[int] = 150
    min_inliers: int = 15
    min_inlier_ratio: float = 0.35
    max_reproj_error: float = 3.0
    det_min_area: int = 4
    det_max_area: int = 2500
    det_diff_thresh: int = 18
    det_max_detections: int = 250
    save_frame_every: int = 5
    step_delay_ms: int = 0


@router.post("")
def create_experiment(req: ExperimentCreateRequest, db: Session = Depends(get_db)):
    uid = f"{datetime.datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
    row = Experiment(
        experiment_uid=uid,
        name=req.name,
        dataset_source=req.dataset_source,
        sequence_name=req.sequence_name,
        trajectory_mode=req.trajectory_mode,
        status="pending",
        params_json=req.dict(),
        output_dir=os.path.join(EXPERIMENTS_ROOT, uid),
    )
    db.add(row)
    db.commit()
    db.refresh(row)
    return dict(experiment_uid=uid, id=row.id)


@router.post("/{uid}/start")
async def start_experiment(uid: str, db: Session = Depends(get_db)):
    row = db.query(Experiment).filter_by(experiment_uid=uid).first()
    if row is None:
        raise HTTPException(404, "experiment not found")
    if row.status == "running":
        return dict(message="already running")

    params = row.params_json

    async def emit(event: dict):
        await manager.broadcast(uid, event)

    asyncio.create_task(run_experiment(uid, params, emit))
    return dict(message="started", experiment_uid=uid)


@router.post("/{uid}/cancel")
def cancel_experiment(uid: str, db: Session = Depends(get_db)):
    row = db.query(Experiment).filter_by(experiment_uid=uid).first()
    if row is None:
        raise HTTPException(404, "experiment not found")
    if row.status != "running":
        return dict(message=f"nothing to cancel (status is '{row.status}')")
    signalled = request_cancel(uid)
    if not signalled:
        return dict(message="experiment is marked running but has no active task "
                             "(likely the server was restarted mid-run) -- "
                             "its status will need to be corrected manually")
    return dict(message="cancel requested -- the run will stop after its current frame "
                         "and save partial results")


@router.get("")
def list_experiments(db: Session = Depends(get_db)):
    rows = db.query(Experiment).order_by(Experiment.created_at.desc()).all()
    return [dict(
        id=r.id, experiment_uid=r.experiment_uid, name=r.name,
        dataset_source=r.dataset_source, sequence_name=r.sequence_name,
        trajectory_mode=r.trajectory_mode, status=r.status,
        created_at=r.created_at, finished_at=r.finished_at,
        results_json=r.results_json,
    ) for r in rows]


@router.get("/{uid}")
def get_experiment(uid: str, db: Session = Depends(get_db)):
    row = db.query(Experiment).filter_by(experiment_uid=uid).first()
    if row is None:
        raise HTTPException(404, "experiment not found")
    return dict(
        id=row.id, experiment_uid=row.experiment_uid, name=row.name,
        dataset_source=row.dataset_source, sequence_name=row.sequence_name,
        trajectory_mode=row.trajectory_mode, status=row.status,
        params_json=row.params_json, results_json=row.results_json,
        output_dir=row.output_dir, error_message=row.error_message,
        created_at=row.created_at, started_at=row.started_at, finished_at=row.finished_at,
    )


@router.delete("/{uid}")
def delete_experiment(uid: str, db: Session = Depends(get_db)):
    row = db.query(Experiment).filter_by(experiment_uid=uid).first()
    if row is None:
        raise HTTPException(404, "experiment not found")
    db.delete(row)
    db.commit()
    return dict(message="deleted")


# ---------------------------------------------------------------------- #
# File-system browsing for a completed/running experiment's output folder
# ---------------------------------------------------------------------- #
@router.get("/{uid}/files")
def list_files(uid: str, db: Session = Depends(get_db)):
    row = db.query(Experiment).filter_by(experiment_uid=uid).first()
    if row is None:
        raise HTTPException(404, "experiment not found")
    store = ExperimentStore(uid)
    return store.list_files()


@router.get("/{uid}/file")
def get_file(uid: str, path: str, db: Session = Depends(get_db)):
    row = db.query(Experiment).filter_by(experiment_uid=uid).first()
    if row is None:
        raise HTTPException(404, "experiment not found")
    store = ExperimentStore(uid)
    full_path = os.path.abspath(os.path.join(store.dir, path))
    if not full_path.startswith(os.path.abspath(store.dir)):
        raise HTTPException(403, "invalid path")
    if not os.path.isfile(full_path):
        raise HTTPException(404, "file not found")
    return FileResponse(full_path)


# ---------------------------------------------------------------------- #
# Live WebSocket stream
# ---------------------------------------------------------------------- #
@router.websocket("/ws/{uid}")
async def experiment_ws(websocket: WebSocket, uid: str):
    await manager.connect(uid, websocket)
    try:
        while True:
            await websocket.receive_text()  # keep-alive / ignored client pings
    except WebSocketDisconnect:
        await manager.disconnect(uid, websocket)
