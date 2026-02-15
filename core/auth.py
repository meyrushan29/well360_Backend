from datetime import datetime, timedelta
from typing import Optional
from jose import JWTError, jwt
from passlib.context import CryptContext

import os
# SECRET CONFIG
# In production, set this via environment variable
SECRET_KEY = os.getenv("JWT_SECRET_KEY", "super_secret_key_change_this_in_production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 3000  # Long expiry for demo

pwd_context = CryptContext(schemes=["pbkdf2_sha256", "bcrypt"], deprecated="auto")

import hashlib

def verify_password(plain_password, hashed_password):
    # Pre-hash with SHA256 to avoid bcrypt 72-byte limit
    safe_pwd = hashlib.sha256(plain_password.encode('utf-8')).hexdigest()
    return pwd_context.verify(safe_pwd, hashed_password)

def get_password_hash(password):
    # Pre-hash with SHA256
    safe_pwd = hashlib.sha256(password.encode('utf-8')).hexdigest()
    return pwd_context.hash(safe_pwd)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt
