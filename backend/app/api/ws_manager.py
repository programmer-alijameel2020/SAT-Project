from __future__ import annotations

from typing import Dict, List
from fastapi import WebSocket
import asyncio
import json


class WSManager:
    def __init__(self):
        self._connections: Dict[str, List[WebSocket]] = {}
        self._lock = asyncio.Lock()

    async def connect(self, experiment_uid: str, ws: WebSocket):
        await ws.accept()
        async with self._lock:
            self._connections.setdefault(experiment_uid, []).append(ws)

    async def disconnect(self, experiment_uid: str, ws: WebSocket):
        async with self._lock:
            conns = self._connections.get(experiment_uid, [])
            if ws in conns:
                conns.remove(ws)

    async def broadcast(self, experiment_uid: str, event: dict):
        conns = self._connections.get(experiment_uid, [])
        dead = []
        payload = json.dumps(event, default=str)
        for ws in conns:
            try:
                await ws.send_text(payload)
            except Exception:
                dead.append(ws)
        for d in dead:
            if d in conns:
                conns.remove(d)


manager = WSManager()
