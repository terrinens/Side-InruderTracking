from fastapi import FastAPI, Depends
from fastapi.responses import StreamingResponse
from fastapi_proj.tracking_apis.mediapipe_api import webcam
from fastapi_proj.database.tables import adminlist
from fastapi_proj.database.database_connection import get_db, Base, Engine
from sqlalchemy.orm import Session

app = FastAPI()
Base.metadata.create_all(Engine)


@app.get("/test/mp/detector")
async def test_mp_detector():
    return StreamingResponse(
        webcam.webcam_person_detector(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


# @app.get("/recode")
# async def test():
#     webcam.webcam_recode()
#     return StreamingResponse(
#         webcam.webcam_person_detector(),
#         media_type="multipart/x-mixed-replace; boundary=frame"
#     )
