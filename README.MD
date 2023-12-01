## IntruderTracking with FastAPI And MediaPipe

### 카메라에 사람이 탐지될 경우 해당 시간을 기록함과 동시에 영상으로 남긴다.

+ 프로젝트에 사용된 요소
    + [FastAPI](https://fastapi.tiangolo.com/ko/)
    + [MediaPipe](https://developers.google.com/mediapipe/solutions)
        + [Object Detection](https://mediapipe-studio.webapps.google.com/demo/object_detector)

## Demo

* 시연 예시, 핵심 코드
    ### 침입자가 감지될시 기록이 시작된다.
    ![Animation1](https://private-user-images.githubusercontent.com/83992590/286826431-02a6f7ae-a5e2-4bd4-9c9c-5c675a406b57.gif?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTEiLCJleHAiOjE3MDEzOTAwMjcsIm5iZiI6MTcwMTM4OTcyNywicGF0aCI6Ii84Mzk5MjU5MC8yODY4MjY0MzEtMDJhNmY3YWUtYTVlMi00YmQ0LTljOWMtNWM2NzVhNDA2YjU3LmdpZj9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFJV05KWUFYNENTVkVINTNBJTJGMjAyMzEyMDElMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjMxMjAxVDAwMTUyN1omWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPWIzOTdhOTdiODk2MjMxZWU4MzkwYjY0YjYxYjY1ZWUyZTdiZDEwOTJmNmJhYzU3NWI5YjRlZTMxNzI4MTg5OGImWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0JmFjdG9yX2lkPTAma2V5X2lkPTAmcmVwb19pZD0wIn0.1G_CuzrdxS6K1NeC-3w0mcGI6feUEol_2SPDgAH0BbI)
    ``` python
    if dc := detect.detections:
        start_time = start_time or time.time()
        count_detector = max(count_detector, len(dc))

        if (time.time() - start_time >= 5) and not person_detected:
            fl = filelist.new_list(now.hour, now.minute)
            save_path = f"{VIDEOS_DIR}\\{fl.date}\\{fl.date}_{fl.time.strftime('%H-%M-%S')}.mp4".strip()
            person_detected = True
  
        end_time = end_time or (time.time() if person_detected else None)
  
        if person_detected:
          queue.put((buffer, detect, save_path))
  
        if end_time and (time.time() - end_time) > 5:
          queue.put(None)
          db.add(fl)
          db.commit()
          queue, writer_process = writer_processing_start()
  
          start_time = end_time = fl, person_detected = None
          person_detected = False

    await websocket.send_bytes(encoded_frame)
    ```
  
    ### Multiprocessing을 이용한 기록 저장 최적화
    ``` python
    if person_detected:
        queue.put((buffer, detect, save_path))
    
    if end_time and (time.time() - end_time) > 5:
        queue.put(None)

        db.add(fl)
        db.commit()

        queue, writer_process = writer_processing_start()

        start_time = end_time = fl, person_detected = None
        person_detected = False
  
  
    ~~~ ~~~ ~~~ ~~~
    writer_process.join()
    ```

