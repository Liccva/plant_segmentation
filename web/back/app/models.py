from sqlalchemy import Column, Integer, String, ForeignKey, Text
from sqlalchemy.orm import relationship
from app.database import Base

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    login = Column(String(50), unique=True, index=True, nullable=False)
    password = Column(String(255), nullable=False)  # Хранить хэш пароля
    role = Column(String(10), nullable=False, default="user")  # user/admin

    # Связь с таблицей предсказаний
    predictions = relationship("Prediction", back_populates="user")

class Prediction(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    picture = Column(Text().with_variant(Text(4294967295), 'mysql'), nullable=False)  # Для MySQL LONGTEXT
    root = Column(Text, nullable=False)     # polygon координаты корня
    stem = Column(Text, nullable=False)     # polygon координаты стебля
    leaf = Column(Text, nullable=False)     # polygon координаты листа

    # Связь с пользователем
    user = relationship("User", back_populates="predictions")