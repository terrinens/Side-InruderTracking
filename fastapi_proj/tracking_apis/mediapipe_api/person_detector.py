import numpy as np
import mediapipe as mp
import cv2

from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from fastapi_proj.tracking_apis.mediapipe_api.Visualizations.person_detecor_Vz import visualize as pvz

# noinspection SpellCheckingInspection
MODEL_FILE = ("C:\\AI-X\\Side Proj\\Side-IntruderTracking\\fastapi_proj\\tracking_apis\\mediapipe_api\\models"
              "\\efficientdet_lite0.tflite")

base_options = python.BaseOptions(model_asset_path=MODEL_FILE)
options = vision.ObjectDetectorOptions(
    base_options=base_options,
    score_threshold=0.5,
    category_allowlist=["person"]
)
detector = vision.ObjectDetector.create_from_options(options)


def private_cv2_img_encode(buffer):
    img = cv2.imdecode(
        np.frombuffer(buffer, np.uint8),
        cv2.IMREAD_COLOR
    )

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    )

    de = detector.detect(mp_image)
    annotated_image = pvz(np.copy(mp_image.numpy_view()), de)

    rgb_annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
    return cv2.imencode(".jpg", rgb_annotated_image)[1]


def get_webcam_detector(buffer):
    img_encode = private_cv2_img_encode(buffer)
    return img_encode.tobytes()


# def get_detector_data(buffer):
#     mp_image = mp.Image(
#         image_format=mp.ImageFormat.SRGB,
#         data=cv2.cvtColor(
#             cv2.imdecode(
#                 np.frombuffer(buffer, np.uint8),
#                 cv2.IMREAD_COLOR
#             ),
#             cv2.COLOR_BGR2RGB
#         )
#     )
#     detector.detect_async(mp_image)
