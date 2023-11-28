from fastapi import FastAPI, Depends, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi_proj.tracking_apis.mediapipe_api import webcam
from fastapi_proj.database.tables import adminlist, filelist
from fastapi_proj.database.database_connection import get_db, Base, Engine
from sqlalchemy.orm import Session

import cv2

app = FastAPI()
Base.metadata.create_all(Engine)


@app.get("/")
async def get_root():
    return HTMLResponse("""
    <html>
    <head>
        <title>Webcam Streaming</title>
    </head>
    <body>
        <h1>Webcam Live Streaming</h1>
        <img id="video" width="1280" height="640">
        <script>
        var video = document.getElementById('video');
        var ws = new WebSocket('ws://localhost:8000/recode');
        ws.onmessage = function(event) {
            var blob = new Blob([event.data], { 'type': 'image/jpeg' });
            var url = URL.createObjectURL(blob);
            video.src = url;
        };
        </script>
    </body>
    </html>
    """)


@app.websocket("/recode")
async def test(websocket: WebSocket, db: Session = Depends(get_db)):
    await webcam.webcam_websocket_recode(websocket, filelist.new_list(), db)
