import os.path

import numpy as np
import mediapipe as mp
import cv2

from fastapi import WebSocket
from websockets.exceptions import ConnectionClosedError
from typing import Optional
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from fastapi_proj.tracking_apis.mediapipe_api.Visualizations.person_detecor_Vz import visualize

# noinspection SpellCheckingInspection
# 모델의 경로를 구성해줍니다.
MODEL_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "models",
    "efficientdet_lite0.tflite"
)

# 'Visualization'에 사용할 옵션을 정의 합니다.
default_options = vision.ObjectDetectorOptions(
    display_names_locale="ko",
    base_options=python.BaseOptions(model_asset_path=MODEL_FILE),
    score_threshold=0.7,
    category_allowlist=["person"]
)

# 정의한 옵션을 'detector'에 주입시켜 새 'detector'를 만듭니다.
default_detector = vision.ObjectDetector.create_from_options(default_options)


def create_mp_image(buffer):
    """
    매개변수 'buffer'를 사용해 Mediapipe Image로 변환하여 리턴하기 위한 메서드 입니다.
    :param buffer: Mediapipe Image로 변환 시키기 위한 매개 변수 입니다.
    :return: 'buffer'를 Mediapipe Image에 맞게 변형 시켜 반환합니다.
    """
    img = cv2.imdecode(
        np.frombuffer(buffer, np.uint8),
        cv2.IMREAD_COLOR
    )

    return mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    )


def get_detect(mp_image: mp.Image):
    """
    'detect'한 이미지를 반환하기 위한 메서드 입니다.
    :param mp_image: 이미지내에 탐지한 객체를 프레임을 씌우고 반환하기 위한 매개 변수입니다.
    :return: 'detect'한 이미지
    """
    return default_detector.detect(mp_image)


def detect_and_convert_color(buffer, detect: default_detector.detect):
    """
    'detect'한 이미지를 cv2 이미지 컬러와 매칭 시키기 위한 메서드 입니다.
    :param buffer: 'create_mp_image()'를 실행 시키기 위한 매개 변수 입니다.
    :param detect: 'detect'한 객체를 'visualize'화 하기 위한 매개 변수 입니다.
    :return: 'visualize'화된 이미지를 cv2로 변환 시킨 이미지
    """

    # ''get_mp_image''를 참조하세요.
    mp_image = create_mp_image(buffer)

    # Mediapipe Image로 변환 시킨 이미지를 'detect'에서 설정된 이미지를 매칭시켜 새 이미지를 생성하기 위함입니다.
    annotated_image = visualize(
        np.copy(mp_image.numpy_view()),
        detect
    )
    return cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)


def get_img_encode_bytes(buffer, detect: Optional[default_detector.detect] = None):
    """
    'detect'된 이미지를 인코딩 하여 반환하기 위한 메서드 입니다.
    :param buffer: 'detect'된 이미지 생성 및 cv2의 인코딩과 매칭시키기 위한 매개 변수입니다.
    :param detect: 'detect_and_convert_color()'를 실행 시키기 위한 매개 변수입니다.
    :return: cv2 Image화된 객체의 bytes를 반환합니다.
    """
    detect = detect or get_detect(create_mp_image(buffer))
    img_encode = cv2.imencode(".jpg", detect_and_convert_color(buffer, detect))[1]
    return img_encode.tobytes()


def get_detect_and_bytes(buffer):
    """
    'detect'된 이미지의 bytes를 생성하기 위해 정의된 메서드들의 총집합입니다.
    :param buffer: 'detect'화 시키고, 컬러를 매칭 시키기 위한 매개 변수입니다.
    :return: 지속적으로 사용될 수 있는 'detect'와 'detect'가 완료된 'encoded_bytes'를 반환합니다.
    """
    detect = get_detect(create_mp_image(buffer))
    encoded_bytes = get_img_encode_bytes(buffer, detect)
    return detect, encoded_bytes


async def ex_webcam_person_detector(websocket: WebSocket, webcam: Optional[cv2.VideoCapture]):
    """
    웹캠을 이용하여 웹소켓으로 지속적으로 송출하기 위한 예시 메서드 입니다.
    :param websocket: 지속적이고 깔끔한 송신을 위한 웹소켓을 이용해 송출하기 위함입니다.
    :param webcam: 촬영을 위한 웹캠입니다.
    :return: 탐지한 이미지를 encode 하여 bytes 를 반환합니다.
    """

    await websocket.accept()

    # 해당 평가문은 웹캠을 접속함과 동시에 조건문을 처리하기 위해 사용됩니다.
    while webcam.isOpened():

        # 웹캠에서 리턴하는 객체들을 사용하기 위해 각 변수에 할당해줍니다.
        success, frame = webcam.read()

        # 웹캠 접속에 실패할 경우 웹캠 자원을 해제하고, 강제로 메서드를 종료합니다.
        if not success:
            webcam.release()
            break

        # 만약 cv2.imshow("x", frame))를 사용해서 웹캠 테스트를 진행할 경우에 사용되는 break 코드 입니다.
        # if cv2.waitKey(1) & 0xFF == ord('q'):
            # break

        # 웹캠 접속에 성공 했을 경우에 진행 됩니다.
        try:

            # 인코딩된 객체를 풀어 사용하기 위해 각 변수에 할당해줍니다.
            _, buffer = cv2.imencode('.jpg', frame)

            # 'buffer'를 사용해 이미지가 인코딩된 객체를 변수에 할당해 줍니다.
            re_frame = get_img_encode_bytes(buffer)

            # 싱크를 맞추기 위해 'await'를 사용하여 웹소켓에 bytes를 보내줍니다.
            await websocket.send_bytes(re_frame)

        # 해당 평가식은 사용자가 강제로 연결을 끊을 경우를 대비한 평가식입니다.
        # 해당 에러에서만 특별히 처리할 객체가 존재하지 않으니 'finally'에서 처리하기 위해 pass를 사용했습니다.
        except ConnectionClosedError:
            pass

        # 모든 코드가 끝난뒤, 필수적으로 자원을 해제하기 위한 평가식입니다.
        finally:
            webcam.release()
