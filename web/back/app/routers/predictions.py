# app/routers/predictions.py
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List

from .. import schemas, crud
from ..database import get_db
from ..auth import get_current_user
from ..models import User

router = APIRouter(prefix="/predictions", tags=["predictions"])

@router.post("/", response_model=schemas.Prediction)
def create_prediction(prediction: schemas.PredictionCreate, db: Session = Depends(get_db)):
    db_user = crud.get_user(db, user_id=prediction.user_id)
    if db_user is None:
        raise HTTPException(status_code=404, detail="User not found")
    return crud.create_prediction(db=db, prediction=prediction)

@router.get("/user/{user_id}", response_model=List[schemas.Prediction])
def read_predictions_by_user(user_id: int, skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    return crud.get_predictions_by_user(db, user_id=user_id, skip=skip, limit=limit)

@router.get("/{prediction_id}", response_model=schemas.Prediction)
def read_prediction(prediction_id: int, db: Session = Depends(get_db)):
    db_prediction = crud.get_prediction(db, prediction_id=prediction_id)
    if db_prediction is None:
        raise HTTPException(status_code=404, detail="Prediction not found")
    return db_prediction

@router.delete("/{prediction_id}")
def delete_prediction(prediction_id: int, db: Session = Depends(get_db)):
    db_prediction = crud.delete_prediction(db, prediction_id=prediction_id)
    if db_prediction is None:
        raise HTTPException(status_code=404, detail="Prediction not found")
    return {"message": "Prediction deleted successfully"}

# Новый эндпоинт для получения истории текущего пользователя
@router.get("/my", response_model=List[schemas.Prediction])
def get_my_predictions(
        skip: int = 0,
        limit: int = 100,
        current_user: User = Depends(get_current_user),
        db: Session = Depends(get_db)
):
    """Возвращает все предсказания текущего пользователя"""
    return crud.get_predictions_by_user(db, user_id=current_user.id, skip=skip, limit=limit)