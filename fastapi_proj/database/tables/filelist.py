from datetime import datetime, time
from typing import Optional, Union

from sqlalchemy import Column, Integer, Date, Time

from fastapi_proj.database.database_connection import Base


class FileList(Base):
    now = datetime.now()
    __tablename__ = 'filelist'
    index = Column(Integer, primary_key=True, autoincrement=True)
    date = Column(Date, default=now.date())
    time = Column(Time, default=now.time())
    count_detector = Column(Integer)

    def set_column_count_detector(self, count: int):
        self.count_detector = count


def new_list(hour: Optional[Union[time, int]] = None, minute: Optional[Union[time, int]] = None,
             second: Optional[Union[time, int]] = None):

    now = datetime.now()

    if not hour:
        hour = now.time().hour
    if isinstance(hour, time):
        hour = hour.hour

    if not minute:
        minute = now.time().minute
    if isinstance(minute, time):
        minute = minute.minute

    if not second:
        second = now.time().second
    if isinstance(second, time):
        second = second.second

    return FileList(
        date=now.date(),
        time=time(hour, minute, second),
        count_detector=1
    )
