from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, JSON, Boolean, Text
from sqlalchemy.orm import relationship
import datetime
from .database import Base

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True)
    password_hash = Column(String)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    
    # Profile Data (Persisted)
    age = Column(Integer, nullable=True)
    gender = Column(String, nullable=True)
    weight = Column(Float, nullable=True)
    height = Column(Float, nullable=True)

    # Relationships
    hydration_entries = relationship("HydrationData", back_populates="owner")
    lip_entries = relationship("LipAnalysis", back_populates="owner")
    mental_health_entries = relationship("MentalHealthAnalysis", back_populates="owner")

class HydrationData(Base):
    __tablename__ = "hydration_data"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)
    
    # Inputs stored as JSON for flexibility
    input_data = Column(JSON)
    
    # Key Results
    recommended_liters = Column(Float)
    risk_level = Column(String)
    
    owner = relationship("User", back_populates="hydration_entries")

class LipAnalysis(Base):
    __tablename__ = "lip_analysis"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)
    
    image_path = Column(String)
    prediction = Column(String)
    hydration_score = Column(Float)
    confidence = Column(Float)
    
    owner = relationship("User", back_populates="lip_entries")

class MentalHealthAnalysis(Base):
    __tablename__ = "mental_health_analysis"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)
    
    emotion = Column(String)
    confidence = Column(Float)
    source = Column(String) # "video", "audio"
    
    # Audio specific (optional)
    tone = Column(String, nullable=True)
    energy = Column(String, nullable=True)
    
    # Video specific (optional)
    faces_detected = Column(Integer, nullable=True)
    
    owner = relationship("User", back_populates="mental_health_entries")

class HydrationSuggestion(Base):
    """
    Personalized suggestions for hydration based on various conditions.
    Supports both Form Prediction and Lip Image Analysis.
    """
    __tablename__ = "hydration_suggestions"

    id = Column(Integer, primary_key=True, index=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow)
    
    # Suggestion Metadata
    title = Column(String, nullable=False)  # Short title (e.g., "Stay Hydrated in Hot Weather")
    content = Column(Text, nullable=False)  # Detailed suggestion text
    category = Column(String, nullable=False)  # "general", "exercise", "weather", "symptoms", "diet", "lifestyle"
    priority = Column(Integer, default=1)  # 1=Low, 2=Medium, 3=High
    is_active = Column(Boolean, default=True)  # Enable/disable suggestions
    
    # Condition Matching (NULL means applies to all)
    # Form Prediction Conditions
    risk_level = Column(String, nullable=True)  # "Low", "Mild Dehydration", "High Dehydration", NULL=all
    min_recommended_liters = Column(Float, nullable=True)  # Minimum recommended water
    max_recommended_liters = Column(Float, nullable=True)  # Maximum recommended water
    activity_level = Column(String, nullable=True)  # "Sedentary", "Light", "Moderate", "Heavy", "Very Heavy"
    temperature_min = Column(Float, nullable=True)  # Minimum temperature (Celsius)
    temperature_max = Column(Float, nullable=True)  # Maximum temperature (Celsius)
    has_symptoms = Column(Boolean, nullable=True)  # If user has any symptoms (thirsty, dizzy, fatigue, headache)
    
    # Lip Analysis Conditions
    lip_prediction = Column(String, nullable=True)  # "Dehydrate", "Normal", NULL=all
    min_hydration_score = Column(Float, nullable=True)  # Minimum hydration score (0-100)
    max_hydration_score = Column(Float, nullable=True)  # Maximum hydration score (0-100)
    
    # Time-based conditions
    time_slots = Column(JSON, nullable=True)  # List of time slots, e.g., ["8 AM-12 PM", "12 PM-4 PM"]
    
    # Target Model
    model_type = Column(String, nullable=False)  # "form", "lip", "both"
