import os
import time
from datetime import datetime
from typing import Optional

import cv2
from fastapi import WebSocket, Depends
from sqlalchemy.orm import Session
from websockets.exceptions import ConnectionClosedError

import fastapi_proj.tracking_apis.mediapipe_api.person_detector as p_det
from fastapi_proj.database.tables import filelist
from fastapi_proj.database.tables.filelist import FileList
from fastapi_proj.database import database_connection as db_c

from multiprocessing import Process, Queue

now = datetime.now()

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


def video_writer(queue: Queue):
    out = None
    while True:
        item = queue.get()
        if item is None:
            break

        frame, detect, save_path = item
        if out is None:
            out = cv2.VideoWriter(save_path, cv2.VideoWriter.fourcc(*'mp4v'), fps, size)

        out.write(p_det.get_cvt_color(frame, detect))

    if out is not None:
        out.release()


def writer_processing_start():
    queue = Queue()
    writer_process = Process(target=video_writer, args=(queue,))
    writer_process.start()
    return queue, writer_process


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


async def webcam_websocket_recode(websocket: WebSocket, db: Session = Depends(db_c.get_db)):
    await websocket.accept()
    queue, writer_process = writer_processing_start()

    save_path = None

    start_time = None
    end_time = None

    fl = None
    person_detected = False
    count_detector = 0

    try:
        while webcam.isOpened():
            frame = read_webcam_frame(webcam)
            _, buffer = cv2.imencode('.jpg', frame)

            detect, encoded_frame = p_det.get_detect_and_bytes(buffer)

            if dc := detect.detections:
                if count_detector < len(dc):
                    count_detector = len(dc)

                if start_time is None:
                    start_time = time.time()

                elif (time.time() - start_time >= 5) and not person_detected:
                    fl = filelist.new_list(now.hour, now.minute)
                    save_path = f"{VIDEOS_DIR}\\{fl.date}\\{fl.date}_{fl.time.strftime('%H-%M-%S')}.mp4".strip()
                    print(fl)
                    print(save_path)
                    person_detected = True

            elif person_detected and not end_time:
                print("엔드타임 시작")
                end_time = time.time()

            if person_detected:
                queue.put((buffer, detect, save_path))

            if end_time:
                print((time.time() - end_time))

            if end_time and (time.time() - end_time) > 5:
                print("5초 이상 감지되지 않음")
                queue.put(None)
                db.add(fl)
                db.commit()

                queue, writer_process = writer_processing_start()

                start_time, end_time, fl = None, None, None
                person_detected = False

            await websocket.send_bytes(encoded_frame)

    except ConnectionClosedError:
        pass
    finally:
        queue.put(None)
        await closable(webcam, None)
        writer_process.join()

    if fl is not None:
        db.add(fl)
        db.commit()
