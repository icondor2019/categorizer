from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from app.configuration.settings import settings

DB_USER = settings.POSTGRES_USER
DB_PASSWORD = settings.POSTGRES_PASSWORD
DB_HOST = settings.POSTGRES_HOST
DB_PORT = settings.POSTGRES_PORT
DB_DATABASE = settings.POSTGRES_DB


def get_engine():
    url = f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_DATABASE}"
    engine = create_engine(url, pool_size=50, echo=False)
    return engine


def get_session() -> Session:
    engine = get_engine()
    session = sessionmaker(bind=engine)
    return session()


db = get_session()
