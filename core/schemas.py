from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime

# Auth Schemas
class RegisterRequest(BaseModel):
    email: str
    password: str

class LoginRequest(BaseModel):
    email: str
    password: str

# Hydration Schemas
class FormPredictionRequest(BaseModel):
    Age: int
    Gender: str
    Weight: float
    Height: float
    Water_Intake_Last_4_Hours: float
    Exercise_Time_Last_4_Hours: float
    Physical_Activity_Level: str
    Urinated_Last_4_Hours: str
    Urine_Color: int
    Thirsty: str
    Dizziness: str
    Fatigue: str
    Headache: str
    Sweating_Level: str
    Time_Slot: str = None  # Optional: User selected slot
    Latitude: float
    Longitude: float

class ImageBase64Request(BaseModel):
    image_base64: str

# Hydration Suggestion Schemas
class HydrationSuggestionCreate(BaseModel):
    """Schema for creating a new hydration suggestion"""
    title: str = Field(..., min_length=1, max_length=200)
    content: str = Field(..., min_length=1)
    category: str = Field(..., description="general, exercise, weather, symptoms, diet, lifestyle")
    priority: int = Field(default=1, ge=1, le=3, description="1=Low, 2=Medium, 3=High")
    is_active: bool = True
    
    # Form Prediction Conditions (Optional)
    risk_level: Optional[str] = None
    min_recommended_liters: Optional[float] = None
    max_recommended_liters: Optional[float] = None
    activity_level: Optional[str] = None
    temperature_min: Optional[float] = None
    temperature_max: Optional[float] = None
    has_symptoms: Optional[bool] = None
    
    # Lip Analysis Conditions (Optional)
    lip_prediction: Optional[str] = None
    min_hydration_score: Optional[float] = None
    max_hydration_score: Optional[float] = None
    
    # Time-based conditions
    time_slots: Optional[List[str]] = None
    
    # Target Model
    model_type: str = Field(..., description="form, lip, or both")

class HydrationSuggestionUpdate(BaseModel):
    """Schema for updating an existing suggestion"""
    title: Optional[str] = None
    content: Optional[str] = None
    category: Optional[str] = None
    priority: Optional[int] = Field(None, ge=1, le=3)
    is_active: Optional[bool] = None
    
    risk_level: Optional[str] = None
    min_recommended_liters: Optional[float] = None
    max_recommended_liters: Optional[float] = None
    activity_level: Optional[str] = None
    temperature_min: Optional[float] = None
    temperature_max: Optional[float] = None
    has_symptoms: Optional[bool] = None
    
    lip_prediction: Optional[str] = None
    min_hydration_score: Optional[float] = None
    max_hydration_score: Optional[float] = None
    
    time_slots: Optional[List[str]] = None
    model_type: Optional[str] = None

class HydrationSuggestionResponse(BaseModel):
    """Schema for returning suggestion data"""
    id: int
    created_at: datetime
    updated_at: datetime
    title: str
    content: str
    category: str
    priority: int
    is_active: bool
    
    risk_level: Optional[str]
    min_recommended_liters: Optional[float]
    max_recommended_liters: Optional[float]
    activity_level: Optional[str]
    temperature_min: Optional[float]
    temperature_max: Optional[float]
    has_symptoms: Optional[bool]
    
    lip_prediction: Optional[str]
    min_hydration_score: Optional[float]
    max_hydration_score: Optional[float]
    
    time_slots: Optional[List[str]]
    model_type: str
    
    class Config:
        from_attributes = True
