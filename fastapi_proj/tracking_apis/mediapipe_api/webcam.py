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

# 영상을 저장할 경로를 지정을 합니다.
VIDEOS_DIR = "C:\\Side Proj\\Side-IntruderTracking\\videos\\"

# 사용할 웹캠을 지정을 합니다.
webcam = cv2.VideoCapture(0)

# 영상 저장을 위한 사이즈를 지정합니다.
# 해당 변수를 예시와 같이 선언하면 웹캠의 해상도에 맞춰 영상이 저장됩니다.
size = (
    int(webcam.get(cv2.CAP_PROP_FRAME_WIDTH)),
    int(webcam.get(cv2.CAP_PROP_FRAME_HEIGHT))
)

# fps 값을 맞추기 위해 웹캠에 fps 값을 가져옵니다.
fps = webcam.get(cv2.CAP_PROP_FPS)


def create_today_folder(target_dir: str):
    """
    'VIDEOS_DIR'에 할당한 폴더에 지금 날짜의 폴더가 존재하는지 확인하고
    존재하지 않을 경우에 자동적으로 생성하기 위한 메서드 입니다.
    앱이 시작될 경우에 실행 됩니다.
    :param target_dir: 자동으로 폴더를 생성할 위치를 지정하기 위한 매개 변수 입니다.
    """
    todaydir = target_dir + str(datetime.now().date())
    if not os.path.isdir(todaydir):
        os.mkdir(todaydir)


create_today_folder(VIDEOS_DIR)


async def closable(
        webcam_: cv2.VideoCapture,
        out: Optional[cv2.VideoWriter] = None,
        queue: Optional[Queue] = None
):
    """
    해당 py 파일에서 사용되는 리소스들을 한번에 해제하기 위한 메서드 입니다.
    :param webcam_: 웹캠 자원
    :param out: 저장하기 위한 자원
    :param queue: 'multiprocessing'을 위한 자원
    """
    if not webcam_.release():
        webcam_.release()
    if out is not None:
        out.release()
    if queue is not None:
        queue.put(None)
    cv2.destroyAllWindows()


def read_webcam_frame(webcam_: webcam):
    """
    웹캠에 접근하기 위한 메서드 입니다.
    성공할 경우에만 웹캠의 frame 값을 반환합니다.
    :param webcam_: 웹캠에 접근 하기 위한 매개 변수입니다.
    :return: frame 값을 반환 합니다.
    """
    success, frame = webcam_.read()
    if not success:
        webcam_.release()
        return None
    return frame


def video_writer(queue: Queue):
    """
    'multiprocessing'을 사용하여 웹캠의 영상을 저장하기 위한 메서드 입니다.
    :param queue: ''multiprocessing'을 사용하기 위한 큐 객체입니다.
    """

    # 지역적으로 사용하기 위한 변수입니다.
    out = None

    while True:

        # 'item' 변수를 선언함과 동시에 평가를 진행하기 위한 코드 입니다.
        # 'queue.get()'에 값이 존재하지 않을 경우에 코드가 중지됩니다.
        if (item := queue.get()) is None:
            break

        # 'item'의 값을 언패킹 하여 사용하기 위해 할당 합니다.
        # 'frame', 'detect', 'save_path'의 변수의 의미는 'queue.get()'으로 넘겨 받은 객체가 해당 객체들로 제한 하겠다는 의미입니다.
        frame, detect, save_path = item

        # 'out'이 ''None''일 경우에 'out'에 영상 기록을 할당합니다.
        # 'None'으로 선언했다가 해당 평가식에서야 실행되는 이유는 'cv2.VideoWriter()'는 할당됨과 동시에 기록이 시작되기 때문입니다.
        out = out or cv2.VideoWriter(save_path, cv2.VideoWriter.fourcc(*'mp4v'), fps, size)

        # 'queue.get()'에서 넘겨 받은 'frame'과 'detect'를 이용하여 'get_cvt_color'에 넘겨 컬러 패턴을 맞추고, 기록 합니다.
        out.write(p_det.get_cvt_color(frame, detect))

    # 해당 평가식에서야 그동안 넘겨받은 이미지를 이용해 하나의 영상으로 기록합니다.
    if out is not None:
        out.release()


def writer_processing_start():
    """
    'multiprocessing'을 시작하고, 영상을 기록하기 위한 메서드입니다.
    :return: 지속적으로 사용될 'queue'와 해당 작업만을 위한 'writer_process'을 반환합니다.
    """
    queue = Queue()

    # 해당 코드에서는 'multiprocessing'을 하기 위한 메서드와 해당 메서드에서 자원을 받을 'queue'을 지정하여 시작 합니다.
    writer_process = Process(target=video_writer, args=(queue,))
    writer_process.start()

    return queue, writer_process


async def webcam_websocket_recode(websocket: WebSocket, db: Session = Depends(db_c.get_db)):
    """
    :param websocket: 'websocket' 통신을 이용해 지속적으로 영상을 보내기 위한 매개 변수 입니다.
    :param db: 영상이 저장됨과 동시에 db에 기록을 남기기 위한 커넥션 매개 변수 입니다.
    :return: 'websocket' 통신으로 이미지를 Frame 으로 변환 시켜 영상처럼 보여줍니다.
    """

    # 웹 소켓 요청을 수락 합니다.
    await websocket.accept()

    # 미리 정의 해둔 'writer_processing_start'를 실행시켜 ''multiprocessing''을 가동해
    # 이미지를 지속적으로 보내면서도 영상을 저장 할 수 있게 준비합니다.
    queue, writer_process = writer_processing_start()

    # 지역적으로 사용하기 위한 변수들의 선언입니다.
    save_path = start_time = end_time = None
    fl, person_detected, count_detector = FileList, False, 0

    try:
        # 웹캠에 접속 시작
        while webcam.isOpened():

            # 성공 여부, 'numpy.dtype'를 반환 시킵니다.
            _, buffer = cv2.imencode('.jpg', read_webcam_frame(webcam))

            # 이미지 인코드를 한 객체를 이용해 감지를 한 객체 및 감지한 이미지를 인코드를 반환합니다.
            detect, encoded_frame = p_det.get_detect_and_bytes(buffer)

            # 'detect.detections 감지한 객체가 있습니다.
            # 즉 감지한 객체가 존재할 경우 ''List''에 값이 존재하니 ''True''를 반환합니다.
            if dc := detect.detections:

                # 감지한 객체가 존재할 경우 타이머를 시작 합니다.
                # 만약 'start_time'이 이미 값이 있다면(''True''), or 연산자는 'start_time'을 반환합니다.
                # 그렇지 않으면 'time.time()'의 값을 'start_time'에 할당합니다.
                # 즉 'start_time'은 초기화 객체 선언에서 ''None''으로 설정되어 있으니 ''False'' 평가를 반환합니다.('time.time()')
                # 그리고 다시 평가식을 실행 했을때 'start_time'은 ''None''이 ''True'' 평가식이 작동되어 객체를 유지 합니다.
                start_time = start_time or time.time()

                # dc의 담겨져 있는 수는 감지된 객체의 수 만큼이니 ''len''을 이용해 카운트 값을 반환합니다.
                # 최대 감지된 인원수를 기록하기 위해 기존값보다 클 경우 반환합니다.
                count_detector = max(count_detector, len(dc))

                # 'start_time' 과 현재 시간이 5초만큼 차이 나며 'detected'가 ''False''인 경우에 평가식이 작동 합니다.
                # 해당 평가식은 5초 이상 객체가 감지될 경우 기록을 시작 합니다.
                if (time.time() - start_time >= 5) and not person_detected:
                    # 데이터베이스에 값을 저장하기 위해 기록이 종료된 시점의 시간을 사용 하여 새로운 데이터 객체를 생성 합니다.
                    fl = filelist.new_list(now.hour, now.minute)

                    # 영상 이름을 테이블 객체의 값과 동일하게 유지하기 위해 테이블 객체에서 값을 꺼내어 파일 이름을 설정 합니다.
                    save_path = f"{VIDEOS_DIR}\\{fl.date}\\{fl.date}_{fl.time.strftime('%H-%M-%S')}.mp4".strip()

                    # 다음 평가식을 위해 탐지 되었음을 확인하기 위해 ''True''를 반환합니다.
                    person_detected = True

            # 'end_time'이 ''False''일 경우이며 'person_detected'가 ''True''일 경우에 'time.time()'을 반환합니다.
            # else ''None''의 평가식은 옳게 설계되어 있다면 절대 작동해서는 안됩니다.
            end_time = end_time or (time.time() if person_detected else None)

            # 'person_detected'가 ''True''일 경우에 이미지를 ''multiprocessing''을 이용하여 처리 합니다.
            # 'queue.put()' 메서드를 사용하여 큐에 데이터를 추가합니다.
            # put에는 처리할 데이터를 넣습니다.
            # 현재의 경우에는 감지된 이미지를(프레임 처리를 위해) 큐에 데이터를 추가합니다.
            if person_detected:
                queue.put((buffer, detect, save_path))

            # 해당 평가식은 'end_time'이 값이 존재 할 경우 이며 '(객체가 탐지되지 않은 총 시간)'의 값이 5초 이상일 경우 작동합니다.
            if end_time and (time.time() - end_time) > 5:
                # 영상 저장을 종료하기 위하여 'queue.put()'에 ''None''을 전달하여 더이상 데이터가 없음을 알려줍니다.
                queue.put(None)

                # db에 데이터를 저장합니다.
                db.add(fl)
                db.commit()

                # 새로운 동영상 저장을 위해 multiprocessing 재시작 합니다.
                queue, writer_process = writer_processing_start()

                # 새롭게 평가식을 시작 하기 위하여 ''None''과 ''False''를 전달 합니다.
                start_time = end_time = fl, person_detected = None
                person_detected = False

            # 현재 코드에서는 이전 평가식들은 전부 잊어버립시다.
            # 이 코드는 객체가 탐지 됐던, 안됐던 웹소켓을 이용하여 웹캠에 송출하기 위한 코드입니다.
            await websocket.send_bytes(encoded_frame)

    # 해당 예외 처리는 사용자가 갑작스럽게 웹 페이지를 나갈 경우에 발생하는 오류를 처리하기 위함입니다.
    # 하지만 지금 이 코드에서는 무조건적으로 웹캠에서 감지된것을 반환해야 하므로 ''pass''로 처리하여 예외를 무시 합니다.
    except ConnectionClosedError:
        pass

    # 사용된 자원들을 해제하기 위한 코드입니다.
    finally:
        await closable(webcam_=webcam, queue=queue)

        # ''multiprocessing''에 전달된 데이터들을 모두 처리 할때까지 기다립니다.
        # 하지만 착각하기 쉬운것이 있습니다. 'queue.join()'과는 해당 '.join'은 비슷한 작업을 실행하나,
        # 'queue.join()'는 'queue'의 모든 작업들이 끝나기 전까지 기다리는것이고
        # 'writer_process.join()'는 'writer_process'에서 처리되고 있는 모든 작업들만 해당됩니다.
        writer_process.join()
