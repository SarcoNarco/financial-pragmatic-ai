from datetime import datetime, timezone

from sqlalchemy import Column, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import relationship

from api.db import Base


def utcnow():
    return datetime.now(timezone.utc)


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String(255), unique=True, index=True, nullable=False)
    password_hash = Column(String(255), nullable=False)
    created_at = Column(DateTime(timezone=True), default=utcnow, nullable=False)

    transcripts = relationship("Transcript", back_populates="user", cascade="all, delete-orphan")


class Transcript(Base):
    __tablename__ = "transcripts"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    text = Column(Text, nullable=False)
    created_at = Column(DateTime(timezone=True), default=utcnow, nullable=False)

    user = relationship("User", back_populates="transcripts")
    analyses = relationship("Analysis", back_populates="transcript", cascade="all, delete-orphan")


class Analysis(Base):
    __tablename__ = "analyses"

    id = Column(Integer, primary_key=True, index=True)
    transcript_id = Column(Integer, ForeignKey("transcripts.id"), nullable=False, index=True)
    signal = Column(String(64), nullable=False)
    prediction = Column(String(64), nullable=False)
    confidence = Column(Float, nullable=False)
    volatility = Column(String(64), nullable=False)
    drivers = Column(Text, nullable=False)
    score = Column(Float, nullable=False)
    created_at = Column(DateTime(timezone=True), default=utcnow, nullable=False)

    transcript = relationship("Transcript", back_populates="analyses")
