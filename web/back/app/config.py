# app/config.py
import os
from dotenv import load_dotenv

load_dotenv()

# Секретный ключ для JWT (обязательно измените в продакшене)
SECRET_KEY = os.getenv("SECRET_KEY", "Kj8mP9$nR2@vL5*qW7!xE4&yU1pupupu(zC3)")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30