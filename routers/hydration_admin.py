"""
Admin endpoints for managing hydration suggestions.
These endpoints allow administrators to create, update, delete, and manage
personalized hydration suggestions stored in the database.
"""
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from typing import List, Optional
from datetime import datetime as dt

from core.database import get_db
from core.models import User, HydrationSuggestion
from core.deps import get_current_user
from core.schemas import (
    HydrationSuggestionCreate,
    HydrationSuggestionUpdate,
    HydrationSuggestionResponse
)

router = APIRouter(
    prefix="/admin/hydration",
    tags=["Hydration Admin"]
)

# =====================================================
# SUGGESTION MANAGEMENT ENDPOINTS
# =====================================================

@router.post("/suggestions", response_model=HydrationSuggestionResponse)
def create_suggestion(
    suggestion: HydrationSuggestionCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Create a new hydration suggestion.
    
    **Required fields:**
    - title: Short descriptive title
    - content: Detailed suggestion text
    - category: general, exercise, weather, symptoms, diet, lifestyle
    - model_type: form, lip, or both
    
    **Optional conditions:**
    - Form prediction: risk_level, recommended_liters, activity_level, temperature, symptoms
    - Lip analysis: lip_prediction, hydration_score
    - Time-based: time_slots (list of time windows)
    """
    try:
        # Validate category
        valid_categories = ["general", "exercise", "weather", "symptoms", "diet", "lifestyle"]
        if suggestion.category not in valid_categories:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid category. Must be one of: {', '.join(valid_categories)}"
            )
        
        # Validate model_type
        valid_model_types = ["form", "lip", "both"]
        if suggestion.model_type not in valid_model_types:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid model_type. Must be one of: {', '.join(valid_model_types)}"
            )
        
        # Create new suggestion
        db_suggestion = HydrationSuggestion(
            title=suggestion.title,
            content=suggestion.content,
            category=suggestion.category,
            priority=suggestion.priority,
            is_active=suggestion.is_active,
            
            # Form conditions
            risk_level=suggestion.risk_level,
            min_recommended_liters=suggestion.min_recommended_liters,
            max_recommended_liters=suggestion.max_recommended_liters,
            activity_level=suggestion.activity_level,
            temperature_min=suggestion.temperature_min,
            temperature_max=suggestion.temperature_max,
            has_symptoms=suggestion.has_symptoms,
            
            # Lip conditions
            lip_prediction=suggestion.lip_prediction,
            min_hydration_score=suggestion.min_hydration_score,
            max_hydration_score=suggestion.max_hydration_score,
            
            # Time conditions
            time_slots=suggestion.time_slots,
            
            # Model type
            model_type=suggestion.model_type
        )
        
        db.add(db_suggestion)
        db.commit()
        db.refresh(db_suggestion)
        
        return db_suggestion
        
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to create suggestion: {str(e)}")


@router.get("/suggestions", response_model=List[HydrationSuggestionResponse])
def get_all_suggestions(
    model_type: Optional[str] = Query(None, description="Filter by model type: form, lip, both"),
    category: Optional[str] = Query(None, description="Filter by category"),
    is_active: Optional[bool] = Query(None, description="Filter by active status"),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Get all hydration suggestions with optional filters.
    """
    try:
        query = db.query(HydrationSuggestion)
        
        # Apply filters
        if model_type:
            query = query.filter(HydrationSuggestion.model_type == model_type)
        if category:
            query = query.filter(HydrationSuggestion.category == category)
        if is_active is not None:
            query = query.filter(HydrationSuggestion.is_active == is_active)
        
        # Order by priority (high to low) and creation date
        suggestions = query.order_by(
            HydrationSuggestion.priority.desc(),
            HydrationSuggestion.created_at.desc()
        ).all()
        
        return suggestions
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch suggestions: {str(e)}")


@router.get("/suggestions/{suggestion_id}", response_model=HydrationSuggestionResponse)
def get_suggestion(
    suggestion_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Get a specific suggestion by ID.
    """
    suggestion = db.query(HydrationSuggestion).filter(
        HydrationSuggestion.id == suggestion_id
    ).first()
    
    if not suggestion:
        raise HTTPException(status_code=404, detail="Suggestion not found")
    
    return suggestion


@router.put("/suggestions/{suggestion_id}", response_model=HydrationSuggestionResponse)
def update_suggestion(
    suggestion_id: int,
    updates: HydrationSuggestionUpdate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Update an existing suggestion.
    Only provided fields will be updated.
    """
    try:
        suggestion = db.query(HydrationSuggestion).filter(
            HydrationSuggestion.id == suggestion_id
        ).first()
        
        if not suggestion:
            raise HTTPException(status_code=404, detail="Suggestion not found")
        
        # Update only provided fields
        update_data = updates.model_dump(exclude_unset=True) if hasattr(updates, "model_dump") else updates.dict(exclude_unset=True)
        
        # Validate category if provided
        if "category" in update_data:
            valid_categories = ["general", "exercise", "weather", "symptoms", "diet", "lifestyle"]
            if update_data["category"] not in valid_categories:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid category. Must be one of: {', '.join(valid_categories)}"
                )
        
        # Validate model_type if provided
        if "model_type" in update_data:
            valid_model_types = ["form", "lip", "both"]
            if update_data["model_type"] not in valid_model_types:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid model_type. Must be one of: {', '.join(valid_model_types)}"
                )
        
        # Apply updates
        for field, value in update_data.items():
            setattr(suggestion, field, value)
        
        suggestion.updated_at = dt.utcnow()
        
        db.commit()
        db.refresh(suggestion)
        
        return suggestion
        
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to update suggestion: {str(e)}")


@router.delete("/suggestions/{suggestion_id}")
def delete_suggestion(
    suggestion_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Delete a suggestion permanently.
    """
    try:
        suggestion = db.query(HydrationSuggestion).filter(
            HydrationSuggestion.id == suggestion_id
        ).first()
        
        if not suggestion:
            raise HTTPException(status_code=404, detail="Suggestion not found")
        
        db.delete(suggestion)
        db.commit()
        
        return {"message": f"Suggestion {suggestion_id} deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to delete suggestion: {str(e)}")


@router.post("/suggestions/{suggestion_id}/toggle")
def toggle_suggestion_status(
    suggestion_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Toggle a suggestion's active status (enable/disable).
    """
    try:
        suggestion = db.query(HydrationSuggestion).filter(
            HydrationSuggestion.id == suggestion_id
        ).first()
        
        if not suggestion:
            raise HTTPException(status_code=404, detail="Suggestion not found")
        
        suggestion.is_active = not suggestion.is_active
        suggestion.updated_at = dt.utcnow()
        
        db.commit()
        db.refresh(suggestion)
        
        status = "enabled" if suggestion.is_active else "disabled"
        return {
            "message": f"Suggestion {suggestion_id} {status}",
            "is_active": suggestion.is_active
        }
        
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to toggle suggestion: {str(e)}")


@router.post("/suggestions/bulk-create", response_model=List[HydrationSuggestionResponse])
def bulk_create_suggestions(
    suggestions: List[HydrationSuggestionCreate],
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Create multiple suggestions at once.
    Useful for initial setup or importing suggestions.
    """
    try:
        created_suggestions = []
        
        for suggestion in suggestions:
            db_suggestion = HydrationSuggestion(
                title=suggestion.title,
                content=suggestion.content,
                category=suggestion.category,
                priority=suggestion.priority,
                is_active=suggestion.is_active,
                
                risk_level=suggestion.risk_level,
                min_recommended_liters=suggestion.min_recommended_liters,
                max_recommended_liters=suggestion.max_recommended_liters,
                activity_level=suggestion.activity_level,
                temperature_min=suggestion.temperature_min,
                temperature_max=suggestion.temperature_max,
                has_symptoms=suggestion.has_symptoms,
                
                lip_prediction=suggestion.lip_prediction,
                min_hydration_score=suggestion.min_hydration_score,
                max_hydration_score=suggestion.max_hydration_score,
                
                time_slots=suggestion.time_slots,
                model_type=suggestion.model_type
            )
            
            db.add(db_suggestion)
            created_suggestions.append(db_suggestion)
        
        db.commit()
        
        # Refresh all created suggestions
        for suggestion in created_suggestions:
            db.refresh(suggestion)
        
        return created_suggestions
        
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to bulk create suggestions: {str(e)}")


@router.get("/suggestions/stats/summary")
def get_suggestions_stats(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Get summary statistics about suggestions.
    """
    try:
        total = db.query(HydrationSuggestion).count()
        active = db.query(HydrationSuggestion).filter(HydrationSuggestion.is_active == True).count()
        inactive = total - active
        
        # Count by model type
        form_count = db.query(HydrationSuggestion).filter(
            HydrationSuggestion.model_type.in_(["form", "both"])
        ).count()
        lip_count = db.query(HydrationSuggestion).filter(
            HydrationSuggestion.model_type.in_(["lip", "both"])
        ).count()
        
        # Count by category
        categories = {}
        for cat in ["general", "exercise", "weather", "symptoms", "diet", "lifestyle"]:
            count = db.query(HydrationSuggestion).filter(
                HydrationSuggestion.category == cat
            ).count()
            categories[cat] = count
        
        # Count by priority
        priorities = {}
        for priority in [1, 2, 3]:
            count = db.query(HydrationSuggestion).filter(
                HydrationSuggestion.priority == priority
            ).count()
            priorities[f"priority_{priority}"] = count
        
        return {
            "total_suggestions": total,
            "active": active,
            "inactive": inactive,
            "model_types": {
                "form": form_count,
                "lip": lip_count
            },
            "categories": categories,
            "priorities": priorities
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get statistics: {str(e)}")
