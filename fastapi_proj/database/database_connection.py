import sqlalchemy
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# noinspection SpellCheckingInspection
DB_URL = 'sqlite:///fastapi_proj/database/IntruderTracking.sqlite3'
Engine = sqlalchemy.create_engine(DB_URL, connect_args={'check_same_thread': False})
SessionLocal = sessionmaker(autoflush=False, bind=Engine)
Base = declarative_base()
Base.metadata.create_all(Engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
