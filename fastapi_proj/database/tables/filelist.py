from datetime import datetime, time

from sqlalchemy import Column, Integer, Date, Time

from fastapi_proj.database.database_connection import Base


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
