import os.path

import numpy as np
import mediapipe as mp
import cv2

from typing import Optional

from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from fastapi_proj.tracking_apis.mediapipe_api.Visualizations.person_detecor_Vz import visualize as pvz

# noinspection SpellCheckingInspection
MODEL_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "models",
    "efficientdet_lite0.tflite"
)

default_options = vision.ObjectDetectorOptions(
    display_names_locale="ko",
    base_options=python.BaseOptions(model_asset_path=MODEL_FILE),
    score_threshold=0.7,
    category_allowlist=["person"]
)
default_detector = vision.ObjectDetector.create_from_options(default_options)


def get_mp_image(buffer):
    img = cv2.imdecode(
        np.frombuffer(buffer, np.uint8),
        cv2.IMREAD_COLOR
    )

    return mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    )


def get_detect(mp_image: mp.Image):
    return default_detector.detect(mp_image)


def get_cvt_color(buffer, detect: default_detector.detect):
    mp_image = get_mp_image(buffer)
    sol_detect = detect

    annotated_image = pvz(
        np.copy(mp_image.numpy_view()),
        sol_detect
    )
    return cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)


def get_img_encode_bytes(buffer, detect: Optional[default_detector.detect] = None):
    if detect is None:
        detect = get_detect(get_mp_image(buffer))

    img_encode = cv2.imencode(".jpg", get_cvt_color(buffer, detect))[1]
    return img_encode.tobytes()


def get_detect_and_bytes(buffer):
    detect = get_detect(get_mp_image(buffer))
    encoded_bytes = get_img_encode_bytes(buffer, detect)
    return detect, encoded_bytes
