# app/crud.py
from sqlalchemy.orm import Session
from . import models, schemas
from passlib.context import CryptContext
import logging

logger = logging.getLogger(__name__)

# Используем pbkdf2_sha256 вместо bcrypt
pwd_context = CryptContext(
    schemes=["pbkdf2_sha256"],
    deprecated="auto"
)

# ВАЖНО: schemes - это метод, вызываем его!
logger.info(f"Используется хешер: {pwd_context.schemes()}")  # <-- Исправлено: добавлены скобки


def verify_password(plain_password, hashed_password):
    """Проверка пароля"""
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password):
    """Хеширование пароля"""
    return pwd_context.hash(password)


# User CRUD
def get_user(db: Session, user_id: int):
    return db.query(models.User).filter(models.User.id == user_id).first()


def get_user_by_login(db: Session, login: str):
    return db.query(models.User).filter(models.User.login == login).first()


def get_users(db: Session, skip: int = 0, limit: int = 100):
    return db.query(models.User).offset(skip).limit(limit).all()


def create_user(db: Session, user: schemas.UserCreate):
    try:
        hashed_password = get_password_hash(user.password)
        logger.info(f"Пароль захеширован для пользователя {user.login}")

        db_user = models.User(
            login=user.login,
            password=hashed_password,
            role=user.role
        )
        db.add(db_user)
        db.commit()
        db.refresh(db_user)
        logger.info(f"Пользователь {user.login} создан с ID {db_user.id}")
        return db_user
    except Exception as e:
        logger.error(f"Ошибка при создании пользователя: {e}")
        db.rollback()
        raise


# Prediction CRUD
def get_prediction(db: Session, prediction_id: int):
    return db.query(models.Prediction).filter(models.Prediction.id == prediction_id).first()


def get_predictions_by_user(db: Session, user_id: int, skip: int = 0, limit: int = 100):
    return db.query(models.Prediction).filter(models.Prediction.user_id == user_id).offset(skip).limit(limit).all()


def create_prediction(db: Session, prediction: schemas.PredictionCreate):
    db_prediction = models.Prediction(
        user_id=prediction.user_id,
        picture=prediction.picture,
        root=prediction.root,
        stem=prediction.stem,
        leaf=prediction.leaf
    )
    db.add(db_prediction)
    db.commit()
    db.refresh(db_prediction)
    return db_prediction


def delete_prediction(db: Session, prediction_id: int):
    db_prediction = db.query(models.Prediction).filter(models.Prediction.id == prediction_id).first()
    if db_prediction:
        db.delete(db_prediction)
        db.commit()
    return db_prediction