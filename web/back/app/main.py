# app/main.py
from fastapi import FastAPI, File, UploadFile, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
import base64
import json
import logging
import os
from datetime import datetime, timedelta

from .database import engine, Base, get_db
from .routers import users, predictions
from . import models, schemas, crud
from .ml.predictor import get_predictor
from .auth import create_access_token, get_current_user
from .config import ACCESS_TOKEN_EXPIRE_MINUTES

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Создание таблиц
logger.info("Создание таблиц в базе данных...")
Base.metadata.create_all(bind=engine)
logger.info("Таблицы созданы успешно!")

app = FastAPI(
    title="Plant Disease Detection API",
    description="API для анализа растений с использованием YOLO модели",
    version="1.0.0"
)

# CORS для React
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Подключение роутеров
app.include_router(users.router)
app.include_router(predictions.router)


@app.get("/")
def root():
    return {"message": "Plant Disease Detection API", "version": "1.0.0", "status": "running"}


@app.get("/health")
def health_check(db: Session = Depends(get_db)):
    try:
        db.execute("SELECT 1")
        db_status = "connected"
        predictor = get_predictor()
        model_status = "loaded" if predictor.analyzer.model else "not loaded"
    except Exception as e:
        db_status = f"disconnected: {str(e)}"
        model_status = "unknown"
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "database": db_status,
        "model": model_status,
        "api_version": "1.0.0"
    }


@app.post("/token", response_model=schemas.Token)
async def login(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = crud.get_user_by_login(db, login=form_data.username)
    if not user or not crud.verify_password(form_data.password, user.password):
        raise HTTPException(
            status_code=401,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(data={"sub": user.login}, expires_delta=access_token_expires)
    return {"access_token": access_token, "token_type": "bearer"}


@app.get("/users/me", response_model=schemas.User)
async def read_users_me(current_user: models.User = Depends(get_current_user)):
    return current_user


@app.post("/analyze")
async def analyze_image(
        file: UploadFile = File(...),
        current_user: models.User = Depends(get_current_user),
        db: Session = Depends(get_db)
):
    """
    Анализ изображения растения и сохранение результата в историю пользователя.
    """
    try:
        logger.info(f"Получен файл для анализа: {file.filename}")

        if not file.content_type or not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")

        contents = await file.read()
        if len(contents) > 10 * 1024 * 1024:
            raise HTTPException(status_code=400, detail="File too large (max 10MB)")

        predictor = get_predictor()
        result = predictor.predict_from_bytes(contents)

        # Подготовка данных для сохранения
        image_base64 = base64.b64encode(contents).decode('utf-8')
        prediction_data = schemas.PredictionCreate(
            user_id=current_user.id,
            picture=image_base64,
            root=json.dumps(result.get('root', {})),
            stem=json.dumps(result.get('stem', {})),
            leaf=json.dumps(result.get('leaf', {}))
        )

        saved_pred = crud.create_prediction(db=db, prediction=prediction_data)
        result['prediction_id'] = saved_pred.id
        logger.info(f"Предсказание сохранено с ID: {saved_pred.id}")

        return {
            "success": True,
            "message": "Анализ выполнен успешно",
            "data": result
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Неожиданная ошибка при анализе: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Дополнительные эндпоинты (опционально)
@app.post("/analyze-base64")
async def analyze_base64(
        data: dict,
        current_user: models.User = Depends(get_current_user),
        db: Session = Depends(get_db)
):
    """Анализ изображения из base64 строки"""
    try:
        base64_image = data.get('image', '')
        if not base64_image:
            raise HTTPException(status_code=400, detail="No image provided")

        try:
            image_bytes = base64.b64decode(base64_image)
        except:
            raise HTTPException(status_code=400, detail="Invalid base64 image")

        predictor = get_predictor()
        result = predictor.predict_from_bytes(image_bytes)

        image_base64 = base64.b64encode(image_bytes).decode('utf-8')
        prediction_data = schemas.PredictionCreate(
            user_id=current_user.id,
            picture=image_base64,
            root=json.dumps(result.get('root', {})),
            stem=json.dumps(result.get('stem', {})),
            leaf=json.dumps(result.get('leaf', {}))
        )

        saved_pred = crud.create_prediction(db=db, prediction=prediction_data)
        result['prediction_id'] = saved_pred.id

        return {"success": True, "data": result}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Ошибка при анализе: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/model-info")
async def get_model_info():
    """Информация о загруженной модели"""
    try:
        predictor = get_predictor()
        analyzer = predictor.analyzer
        return {
            "status": "loaded",
            "model_type": "YOLO",
            "classes": analyzer.class_names,
            "pixels_per_mm": analyzer.pixels_per_mm
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}


@app.get("/debug/tables")
async def check_tables(db: Session = Depends(get_db)):
    """Проверка таблиц в БД"""
    from sqlalchemy import inspect
    inspector = inspect(engine)
    tables = inspector.get_table_names()
    tables_info = {}
    for table in tables:
        columns = inspector.get_columns(table)
        tables_info[table] = [{"name": col["name"], "type": str(col["type"])} for col in columns]
    return {"tables": tables, "tables_info": tables_info}