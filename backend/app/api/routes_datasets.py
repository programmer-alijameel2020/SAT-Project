from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from ..datasets.viso import VISOLoader
from ..datasets.sdmcar import SDMCarLoader
from ..datasets.demo_synth import DemoSyntheticLoader

router = APIRouter(prefix="/api/datasets", tags=["datasets"])

LOADERS = {"VISO": VISOLoader, "SDM-Car": SDMCarLoader, "demo": DemoSyntheticLoader}


class ScanRequest(BaseModel):
    dataset_source: str
    dataset_root: str = ""


@router.post("/scan")
def scan_dataset(req: ScanRequest):
    if req.dataset_source not in LOADERS:
        raise HTTPException(400, f"unknown dataset_source '{req.dataset_source}'")
    loader_cls = LOADERS[req.dataset_source]
    loader = loader_cls(req.dataset_root) if req.dataset_source != "demo" else loader_cls()
    sequences = loader.list_sequences()
    return dict(
        dataset_source=req.dataset_source,
        dataset_root=req.dataset_root,
        n_sequences=len(sequences),
        sequences=[dict(name=s.name, n_frames=s.n_frames, has_annotations=s.has_annotations)
                   for s in sequences],
    )
