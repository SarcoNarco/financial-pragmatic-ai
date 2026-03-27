from typing import Optional

from pydantic import BaseModel, Field


class AuthRequest(BaseModel):
    email: str
    password: str = Field(min_length=6, max_length=256)


class AuthResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


class TranscriptRequest(BaseModel):
    transcript: str


class SaveAnalysisRequest(BaseModel):
    transcript: str


class CompareRequest(BaseModel):
    transcript_1: Optional[str] = None
    transcript_2: Optional[str] = None
    analysis_id_1: Optional[int] = None
    analysis_id_2: Optional[int] = None
