import json
from typing import Optional, List
from fastapi import WebSocket, WebSocketDisconnect
import asyncio

class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def broadcast(self, message: dict):
        data = json.dumps(message)
        for ws in list(self.active_connections):
            try:
                await ws.send_text(data)
            except WebSocketDisconnect:
                self.disconnect(ws)

manager = ConnectionManager()

async def websocket_endpoint(websocket: WebSocket):
    # 1) 클라이언트 연결 수락 & 저장
    await manager.connect(websocket)
    try:
        # 2) 연결 유지 (클라이언트 메시지 무시)
        while True:
            await asyncio.sleep(1)
    except WebSocketDisconnect:
        manager.disconnect(websocket)

async def notify_progress(
    task_name: str,
    progress: int,
    error: Optional[str] = None
):
    """
    전체 WS 구독자에게 아래 포맷으로 푸시됩니다.
    {
      "task_name": "...",
      "progress": 0|25|50|75|100|-1,
      "status": "running"|"completed"|"failed",
      "error": "..."  # optional
    }
    """
    status = (
        "failed"    if progress < 0 else
        "running"   if progress < 100 else
        "completed"
    )
    payload = {
        "task_name": task_name,
        "progress": progress,
        "status": status,
    }
    if error:
        payload["error"] = error

    await manager.broadcast(payload)
