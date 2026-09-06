from __future__ import annotations

import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from .db.database import init_db
from .api.routes_experiments import router as experiments_router
from .api.routes_datasets import router as datasets_router
from .api.routes_fs import router as fs_router

app = FastAPI(
    title="Satellite MOT Research System",
    description=(
        "Real-Time Motion-Geometry-Grounded Multi-Object Tracking for "
        "Occlusion Recovery in Satellite Video -- experiment backend."
    ),
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(experiments_router)
app.include_router(datasets_router)
app.include_router(fs_router)


@app.on_event("startup")
def on_startup():
    init_db()


@app.get("/api/health")
def health():
    return {"status": "ok"}


# Serve the frontend directly from the backend for a one-command demo.
FRONTEND_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "frontend")
)
if os.path.isdir(FRONTEND_DIR):
    app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
