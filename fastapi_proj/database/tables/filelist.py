from sqlalchemy import Column, Integer, Date, Time, String
from fastapi_proj.database.database_connection import Base


class FileList(Base):
    __tablename__ = 'filelist'
    index = Column(Integer, primary_key=True, autoincrement=True)
    date = Column(Date)
    time = Column(Time)
    count_recode_time = Column(String)
