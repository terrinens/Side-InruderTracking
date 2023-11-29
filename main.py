from fastapi import FastAPI, Depends, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse, HTMLResponse, FileResponse
from fastapi_proj.tracking_apis.mediapipe_api import webcam
from fastapi_proj.database.tables import adminlist, filelist
from fastapi_proj.database.database_connection import get_db, Base, Engine
from sqlalchemy.orm import Session


app = FastAPI()
Base.metadata.create_all(Engine)


@app.get("/", response_class=HTMLResponse)
async def get_root():
    return FileResponse('fastapi_proj/templates/index.html')


@app.websocket("/recode")
async def test(websocket: WebSocket, db: Session = Depends(get_db)):
    await webcam.webcam_websocket_recode(websocket, db)
