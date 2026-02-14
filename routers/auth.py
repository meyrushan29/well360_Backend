from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
from datetime import timedelta

from core.database import get_db
from core.models import User
import core.auth as auth
from core.deps import get_current_user
from core.schemas import RegisterRequest, LoginRequest

router = APIRouter(
    prefix="/auth",
    tags=["Authentication"]
)

@router.post("/register")
def register(request: RegisterRequest, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == request.email).first()
    if user:
        raise HTTPException(status_code=400, detail="Email already registered")
    
    hashed_pw = auth.get_password_hash(request.password)
    new_user = User(email=request.email, password_hash=hashed_pw)
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    return {"message": "User created successfully"}

@router.post("/login")
def login(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    # Compatible with Swagger UI
    user = db.query(User).filter(User.email == form_data.username).first()
    if not user or not auth.verify_password(form_data.password, user.password_hash):
        raise HTTPException(status_code=401, detail="Incorrect email or password")
    
    access_token_expires = timedelta(minutes=auth.ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = auth.create_access_token(
        data={"sub": user.email}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}

# Custom JSON Login for App
@router.post("/login-json")
def login_json(request: LoginRequest, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == request.email).first()
    if not user or not auth.verify_password(request.password, user.password_hash):
        raise HTTPException(status_code=401, detail="Incorrect email or password")
    
    access_token_expires = timedelta(minutes=auth.ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = auth.create_access_token(
        data={"sub": user.email}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer", "user_id": user.id, "email": user.email}

@router.get("/profile")
def get_profile(current_user: User = Depends(get_current_user)):
    return {
        "email": current_user.email,
        "age": current_user.age,
        "gender": current_user.gender,
        "weight": current_user.weight,
        "height": current_user.height
    }

# =====================================================
# GOOGLE AUTHENTICATION
# =====================================================
from pydantic import BaseModel
import requests

class GoogleLoginRequest(BaseModel):
    id_token: str

@router.post("/google")
def google_login(request: GoogleLoginRequest, db: Session = Depends(get_db)):
    try:
        # Verify the token using Google's public keys
        # We can use google-auth library, or simply verify against the tokeninfo endpoint for simplicity in MVP
        # Using endpoint is slower but easier as it handles key rotation and audience checks loosely for now
        # Production should use: google.oauth2.id_token.verify_oauth2_token
        
        # Verify token with Google
        response = requests.get(f"https://oauth2.googleapis.com/tokeninfo?id_token={request.id_token}")
        
        if response.status_code != 200:
             raise HTTPException(status_code=401, detail="Invalid Google Token")
             
        google_data = response.json()
        email = google_data.get("email")
        
        if not email:
            raise HTTPException(status_code=400, detail="Google Account has no email")

        # Check if user exists
        user = db.query(User).filter(User.email == email).first()
        
        if not user:
            # Create new user automatically
            # We set a placeholder password that cannot be used for normal login (starts with ! or similar)
            dummy_hash = auth.get_password_hash(f"GOOGLE_LOGIN_{email}_{auth.SECRET_KEY}")
            
            new_user = User(email=email, password_hash=dummy_hash)
            # Initialize default values if needed, e.g. age=0
            db.add(new_user)
            db.commit()
            db.refresh(new_user)
            user = new_user
            
        # Generate JWT for our app
        access_token_expires = timedelta(minutes=auth.ACCESS_TOKEN_EXPIRE_MINUTES)
        access_token = auth.create_access_token(
            data={"sub": user.email}, expires_delta=access_token_expires
        )
        
        return {
            "access_token": access_token, 
            "token_type": "bearer", 
            "user_id": user.id, 
            "email": user.email,
            "is_new_user": user.age is None # Flag to frontend to show onboarding
        }
        
    except Exception as e:
        print(f"Google Login Error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
