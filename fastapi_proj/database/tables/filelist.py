from datetime import datetime, time
from typing import Optional, Union

from sqlalchemy import Column, Integer, Date, Time

from fastapi_proj.database.database_connection import Base

now = datetime.now()


class FileList(Base):
    __tablename__ = 'filelist'
    index = Column(Integer, primary_key=True, autoincrement=True)
    date = Column(Date, default=now.date())
    time = Column(Time, default=now.time())
    count_detector = Column(Integer)

    def set_column_count_detector(self, count: int):
        self.count_detector = count


def new_list(hour: Optional[Union[time, int]] = None, minute: Optional[Union[time, int]] = None):
    if _ := hour:
        hour = now.time()
    if isinstance(hour, time):
        hour = hour.hour

    if _ := minute:
        minute = now.time()
    if isinstance(minute, time):
        minute = minute.minute

    return FileList(
        time=time(hour, minute),
        count_detector=1
    )
