# app/schemas.py (дополните существующий файл)
from pydantic import BaseModel
from typing import Optional, List


# User schemas
class UserBase(BaseModel):
    login: str
    role: str


class UserCreate(UserBase):
    password: str


class User(UserBase):
    id: int

    class Config:
        from_attributes = True


# Prediction schemas
class PredictionBase(BaseModel):
    user_id: int
    picture: str
    root: str
    stem: str
    leaf: str


class PredictionCreate(PredictionBase):
    pass


class Prediction(PredictionBase):
    id: int

    class Config:
        from_attributes = True


# Token schemas
class Token(BaseModel):
    access_token: str
    token_type: str


class TokenData(BaseModel):
    username: Optional[str] = None