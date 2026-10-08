from typing import List
from fastapi import APIRouter, Depends, status
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.modules.identity.schemas import UserCreate, UserResponse
from app.modules.identity import service

router = APIRouter(prefix="/identity", tags=["Identity & Accounts"])

@router.get("/users/limit", response_model=List[UserResponse])
def read_users(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    return service.get_users(db, skip=skip, limit=limit)

@router.get("/users",response_model=List[UserResponse])
def get_all_users(db:Session=Depends(get_db)):
    all_users=service.get_all_users(db)
    if not all_users:
        raise HTTPException(status_code=404, detail="User not found")
    return all_users

@router.get("/users/{user_id}",response_model=UserResponse)
def read_user_id(user_id:int,db: Session = Depends(get_db)):
    db_user=service.get_users_id(db,user_id=user_id)
    if db_user is None:
        raise HTTPException(status_code=404, detail="User not found")
    return db_user

@router.post("/users", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
def create_new_user(user: UserCreate, db: Session = Depends(get_db)):
    return service.create_user(db=db, user_data=user)