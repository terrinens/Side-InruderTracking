import asyncio
import cv2
import os
import time
import fastapi_proj.tracking_apis.mediapipe_api.person_detector as p_det

from sqlalchemy.orm import Session
from datetime import datetime

from websockets.exceptions import ConnectionClosedError
from fastapi import WebSocketException, WebSocket, Depends
from typing import Generator, AsyncGenerator, AsyncIterable, Optional
from fastapi_proj.database.tables import filelist

VIDEOS_DIR = "C:\\Side Proj\\Side-IntruderTracking\\videos\\"
webcam = cv2.VideoCapture(0)

size = (
    int(webcam.get(cv2.CAP_PROP_FRAME_WIDTH)),
    int(webcam.get(cv2.CAP_PROP_FRAME_HEIGHT))
)
fps = webcam.get(cv2.CAP_PROP_FPS)


def create_today_folder(target_dir: str):
    todaydir = target_dir + str(datetime.now().date())
    if not os.path.isdir(todaydir):
        os.mkdir(todaydir)


create_today_folder(VIDEOS_DIR)


async def closable(
        webcam_: cv2.VideoCapture,
        out: Optional[cv2.VideoWriter] = None
):
    if not webcam_.release():
        webcam_.release()
    if out is not None:
        out.release()
    cv2.destroyAllWindows()


def read_webcam_frame(webcam_: webcam):
    success, frame = webcam_.read()
    if not success:
        webcam_.release()
        return None
    return frame


async def webcam_person_detector(websocket: WebSocket):
    await websocket.accept()
    while webcam.isOpened():
        success, frame = webcam.read()
        if not success:
            webcam.release()
            break

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        try:
            _, buffer = cv2.imencode('.jpg', frame)
            re_frame = p_det.get_img_encode_bytes(buffer)
            await websocket.send_bytes(re_frame)

        except ConnectionClosedError:
            webcam.release()

    await closable(webcam, None)


async def webcam_websocket_recode(websocket: WebSocket, fl: filelist.FileList, db: Session = Depends(filelist.get_db)):
    save_path = f"{VIDEOS_DIR}\\{fl.date}\\{fl.date}_{fl.time.strftime('%H-%M')}.mp4".strip()
    await websocket.accept()

    start_time = None
    person_detected = False
    out = None

    try:
        while webcam.isOpened():
            frame = read_webcam_frame(webcam)
            _, buffer = cv2.imencode('.jpg', frame)

            detect, encoded_frame = p_det.get_detect_and_bytes(buffer)

            dc = detect.detections
            if dc:
                if fl.count_detector < len(dc):
                    filelist.FileList.set_column_count_detector(fl, len(dc))

                if start_time is None:
                    start_time = time.time()
                elif (time.time() - start_time >= 15) and not person_detected:
                    out = cv2.VideoWriter(save_path, cv2.VideoWriter.fourcc(*'mp4v'), fps, size)
                    person_detected = True

            if person_detected:
                out.write(p_det.get_cvt_color(buffer, detect))

            await websocket.send_bytes(encoded_frame )

    except ConnectionClosedError:
        webcam.release()

    if person_detected:
        out.release()
    await closable(webcam, None)

    db.add(fl)
    db.commit()
