from fastapi_proj.database.database_connection import Base, get_db

from fastapi import Depends, WebSocket

from sqlalchemy import Column, Integer, Date, Time, String, func
from sqlalchemy.orm import Session

from datetime import datetime, time

from fastapi_proj.tracking_apis.mediapipe_api import webcam
from fastapi_proj.tracking_apis.mediapipe_api.person_detector import default_detector


class FileList(Base):
    __tablename__ = 'filelist'
    index = Column(Integer, primary_key=True, autoincrement=True)
    date = Column(Date)
    time = Column(Time)
    count_detector = Column(Integer)

    def set_column_count_detector(self, count: int):
        self.count_detector = count


def new_list():
    now = datetime.now()
    return FileList(
        date=now.date(),
        time=time(now.time().hour, now.time().minute),
        count_detector=1
    )
