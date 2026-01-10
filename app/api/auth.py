# app/api/auth.py
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_db
from app.db.models.user import User
from app.services.auth_service import auth_service, get_current_user

router = APIRouter(prefix="/auth", tags=["auth"])


class RegisterRequest(BaseModel):
    login: str = Field(min_length=3, max_length=64)
    password: str = Field(min_length=6, max_length=128)


class LoginRequest(BaseModel):
    login: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


class UserOut(BaseModel):
    id: int
    login: str


@router.post("/register", response_model=UserOut, status_code=201)
async def register(payload: RegisterRequest, db: AsyncSession = Depends(get_db)):
    try:
        user = await auth_service.register(db, payload.login, payload.password)
        return UserOut(id=user.id, login=user.login)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/login", response_model=TokenResponse)
async def login(payload: LoginRequest, db: AsyncSession = Depends(get_db)):
    try:
        token = await auth_service.login(db, payload.login, payload.password)
        return TokenResponse(access_token=token)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=str(e))


@router.get("/me", response_model=UserOut)
async def me(user: User = Depends(get_current_user)):
    return UserOut(id=user.id, login=user.login)
