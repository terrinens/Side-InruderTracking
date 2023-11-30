from sqlalchemy import Column, Integer, String

from fastapi_proj.database.database_connection import Base


class AdminList(Base):
    __tablename__ = 'admin-list'
    index = Column(Integer, primary_key=True, autoincrement=True)
    id = Column(String)
    pwd = Column(String)
    phone = Column(String)
