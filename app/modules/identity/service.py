from sqlalchemy.orm import Session
from app.modules.identity.models import User
from app.modules.identity.schemas import UserCreate

def get_all_users(db:Session):
    return db.query(User).all()
    
def get_users(db: Session, skip: int = 0, limit: int = 100):
    return db.query(User).offset(skip).limit(limit).all()

def get_users_id(db:Session,user_id:int):
    return db.query(User).filter(User.id==user_id).first()

def create_user(db: Session, user_data: UserCreate):
    db_user = User(name=user_data.name, email=user_data.email)
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user