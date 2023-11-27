import cv2
import fastapi_proj.tracking_apis.mediapipe_api.person_detector as p_detector
import os

webcam = cv2.VideoCapture(0)


def webcam_open():
    while webcam.isOpened():
        success, _ = webcam.read()
        if not success:
            webcam.release()
            break
        if cv2.waitKey(1) & 0xFF == ord('q'):
            webcam.release()
            cv2.destroyAllWindows()
            break


def webcam_person_detector():
    webcam_open()
    _, frame = webcam.read()
    _, buffer = cv2.imencode('.jpg', frame)
    re_frame = p_detector.get_webcam_detector(buffer)
    yield (b'--frame\r\n'
           b'Content-Type: image/jpeg\r\n\r\n' + re_frame + b'\r\n')


def webcam_recode():
    size = (
        int(webcam.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(webcam.get(cv2.CAP_PROP_FRAME_HEIGHT))
        )

    out = cv2.VideoWriter(
        "C:\\AI-X\\Side Proj\\Side-IntruderTracking\\videos\\TestVideo.mp4",
        cv2.VideoWriter.fourcc(*'mp4v'),
        25.40,
        size
    )

    while webcam.isOpened():
        success, frame = webcam.read()
        if not success:
            webcam.release()
            break

        out.write(frame)

        if cv2.waitKey(int(1000/25.40)) != -1 or 0xFF == ord('q'):
            webcam.release()
            break

    out.release()
    cv2.destroyAllWindows()
