from sqlalchemy import create_engine, Column, Integer, Float, String, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker
from datetime import datetime

engine = create_engine("sqlite:///tracking.db", echo=False)
Base = declarative_base()
SessionLocal = sessionmaker(bind=engine)

class Detection(Base):
    __tablename__ = "detections"

    id        = Column(Integer, primary_key=True)
    person_id = Column(Integer)
    frame     = Column(Integer)
    action    = Column(String)
    x         = Column(Float)
    y         = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)

def init_db():
    Base.metadata.create_all(bind=engine)
def reset_db():
    Base.metadata.drop_all(bind=engine)
