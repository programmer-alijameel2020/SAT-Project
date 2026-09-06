from __future__ import annotations

import datetime
from sqlalchemy import (
    Column, Integer, String, Float, Boolean, DateTime, Text, JSON, ForeignKey
)
from sqlalchemy.orm import relationship, declarative_base

Base = declarative_base()


class Experiment(Base):
    __tablename__ = "experiments"

    id = Column(Integer, primary_key=True, autoincrement=True)
    experiment_uid = Column(String(64), unique=True, nullable=False)   # folder name too
    name = Column(String(255), nullable=False)
    dataset_source = Column(String(32), nullable=False)   # VISO | SDM-Car | demo
    sequence_name = Column(String(255))
    trajectory_mode = Column(String(32))                  # raw | affine | homography
    status = Column(String(32), default="pending")        # pending|running|completed|failed
    params_json = Column(JSON)                             # full parameter set from frontend
    results_json = Column(JSON)                             # tracker.summary() output
    output_dir = Column(String(512))
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    started_at = Column(DateTime, nullable=True)
    finished_at = Column(DateTime, nullable=True)
    error_message = Column(Text, nullable=True)

    events = relationship("ExperimentEvent", back_populates="experiment",
                           cascade="all, delete-orphan")


class ExperimentEvent(Base):
    """Append-only log of every pipeline stage event, kept for replay in
    the frontend dashboard even after the run finishes."""
    __tablename__ = "experiment_events"

    id = Column(Integer, primary_key=True, autoincrement=True)
    experiment_id = Column(Integer, ForeignKey("experiments.id"))
    seq = Column(Integer)                    # monotonically increasing per-experiment
    stage = Column(String(64))               # e.g. "loading_dataset", "frame_processing"
    payload_json = Column(JSON)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)

    experiment = relationship("Experiment", back_populates="events")
