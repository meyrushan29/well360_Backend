from fastapi import APIRouter, Depends, HTTPException, File, UploadFile, Form
from sqlalchemy.orm import Session
from datetime import datetime
import datetime as dt_module # to allow datetime.datetime and datetime.timezone
import os
import uuid
import base64
import json

from core.database import get_db
from core.models import User, HydrationData, LipAnalysis
from core.deps import get_current_user
from core.schemas import FormPredictionRequest, ImageBase64Request
from core.utils import to_system_local, parse_slot_hours, fetch_personalized_suggestions

# Lazy import inside functions or top level if safe
from hydration.predict_Regression import AdvancedPredictor, get_current_weather, get_current_time_slot
# Lazy import for lip prediction (as in main.py)

router = APIRouter(
    tags=["Hydration"]
)

predictor = AdvancedPredictor()

# =====================================================
# HEALTH CHECK (no auth - for connectivity debugging)
# =====================================================
@router.get("/hydration/health")
def hydration_health():
    """No auth. Use this to verify backend is reachable before calling /predict/lip."""
    from pathlib import Path
    from core.config import MOBILENET_MODEL_OUT
    lip_available = Path(MOBILENET_MODEL_OUT).exists()
    return {
        "status": "ok",
        "module": "hydration",
        "lip_model_available": lip_available,
        "predict_lip_endpoint": "/predict/lip",
    }

# =====================================================
# ROUTES: PREDICTION & WEATHER
# =====================================================

@router.get("/weather/current")
def get_weather(lat: float, lon: float, current_user: User = Depends(get_current_user)):
    try:
        temp, hum = get_current_weather(lat, lon)
        return {
            "temperature_c": temp,
            "humidity_percent": hum,
            "location": {"lat": lat, "lon": lon}
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Weather fetch failed: {e}")

@router.post("/predict/form")
def predict_form(
    data: FormPredictionRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    try:
        print(f"Prediction for User: {current_user.email}")

        user_input = data.model_dump() if hasattr(data, "model_dump") else data.dict()
        user_input["Existing Diseases / Medical Conditions"] = "None"

        # Get Weather
        temp, hum = get_current_weather(data.Latitude, data.Longitude)
        user_input["Temperature_C"] = temp
        user_input["Humidity_%"] = hum

        # Map Pydantic Keys to ML Model Keys
        mapped_input = {
            "Age": user_input["Age"],
            "Gender": user_input["Gender"],
            "Weight": user_input["Weight"], 
            "Height": user_input["Height"], 
            "Water_Intake_Last_4_Hours": user_input["Water_Intake_Last_4_Hours"], 
            "Exercise Time (minutes) in Last 4 Hours": user_input["Exercise_Time_Last_4_Hours"], 
            "Physical_Activity_Level": user_input["Physical_Activity_Level"], 
            "Urinated (Last 4 Hours)": user_input["Urinated_Last_4_Hours"], 
            "Urine Color (Most Recent Urination)": user_input["Urine_Color"], 
            "Thirsty (Right Now)": user_input["Thirsty"], 
            "Dizziness (Right Now)": user_input["Dizziness"], 
            "Fatigue / Tiredness (Right Now)": user_input["Fatigue"], 
            "Headache (Right Now)": user_input["Headache"], 
            "Sweating Level (Last 4 Hours)": user_input["Sweating_Level"], 
            "Time Slot (Select Your Current 4-Hour Window)": user_input.get("Time_Slot") or get_current_time_slot(),
            "Existing Diseases / Medical Conditions": "None",
            "Temperature_C": temp,
            "Humidity_%": hum
        }

        # Run Prediction
        try:
             result = predictor.predict(mapped_input)
        except Exception as pred_err:
             import traceback
             traceback.print_exc()
             raise pred_err

        recs = result["recommendations"]

        # Risk Level Logic
        rec_water = result["hydration_prediction"]["recommended_water_liters_next_4h"]
        risk = "Normal"
        if rec_water > 2.0: risk = "High Dehydration"
        elif rec_water > 1.0: risk = "Mild Dehydration"

        # DELETE OLD ENTRIES FOR SAME TIME SLOT TODAY
        now_local = datetime.now()
        today_date_local = now_local.date()
        
        time_slot = mapped_input.get("Time Slot (Select Your Current 4-Hour Window)", "Unknown")
        
        start_query = datetime.utcnow() - dt_module.timedelta(hours=30)
        
        existing_entries = db.query(HydrationData).filter(
            HydrationData.user_id == current_user.id,
            HydrationData.timestamp >= start_query
        ).all()
        
        for entry in existing_entries:
            entry_local_dt = entry.timestamp.replace(tzinfo=dt_module.timezone.utc).astimezone() if entry.timestamp.tzinfo is None else entry.timestamp.astimezone()
            entry_date_local = entry_local_dt.date()
            
            if entry_date_local == today_date_local and entry.input_data:
                entry_slot = entry.input_data.get("Time Slot (Select Your Current 4-Hour Window)", "")
                if entry_slot == time_slot:
                    db.delete(entry)
        
        # SAVE NEW ENTRY TO DATABASE
        db_entry = HydrationData(
            user_id=current_user.id,
            input_data=user_input,
            recommended_liters=rec_water,
            risk_level=risk
        )
        
        # AUTO-UPDATE PROFILE
        current_user.age = mapped_input["Age"]
        current_user.gender = mapped_input["Gender"]
        current_user.weight = mapped_input["Weight"]
        current_user.height = mapped_input["Height"]
        
        db.add(db_entry)
        db.commit()

        # Fetch personalized suggestions from database
        prediction_context = {
            "risk_level": risk,
            "recommended_liters": rec_water,
            "activity_level": mapped_input["Physical_Activity_Level"],
            "temperature_c": temp,
            "has_symptoms": any([
                mapped_input["Thirsty (Right Now)"] == "Yes",
                mapped_input["Dizziness (Right Now)"] == "Yes",
                mapped_input["Fatigue / Tiredness (Right Now)"] == "Yes",
                mapped_input["Headache (Right Now)"] == "Yes"
            ]),
            "time_slot": mapped_input.get("Time Slot (Select Your Current 4-Hour Window)", "Unknown")
        }
        
        personalized_suggestions = fetch_personalized_suggestions(db, "form", prediction_context)

        return {
            "success": True,
            "recommended_total_water_liters": rec_water,
            "hydration_score": result["hydration_prediction"]["hydration_score"],
            "predicted_medical_conditions": result["disease_risk_profile"],
            "temperature_c": temp,
            "humidity_percent": hum,
            "ai_reasoning": result["hydration_prediction"].get("ai_reasoning", []),
            "recommendations": recs,
            "personalized_suggestions": personalized_suggestions  # NEW: DB-based suggestions
        }
    except Exception as e:
        print(f"PREDICT ERROR: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/predict/lip")
def predict_lip(
    data: ImageBase64Request,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    try:
        from hydration.imagePredict_mobilenet import predict_single

        os.makedirs("temp", exist_ok=True)
        temp_filename = f"temp/{uuid.uuid4()}.png"
        
        img_str = data.image_base64
        if "," in img_str:
            img_str = img_str.split(",")[1]
            
        with open(temp_filename, "wb") as f:
            f.write(base64.b64decode(img_str))
            
        result = predict_single(temp_filename)

        if "error" in result:
             raise HTTPException(status_code=400, detail=result["error"])
        
        # === STORAGE UPDATE: Upload to Firebase if configured ===
        from core.storage import storage_manager

        # Original Hydration Lip Image
        firebase_url = storage_manager.upload_file(
            result["saved_image_path"], 
            remote_folder="hydration_images",
            local_url_prefix="/uploads",
            content_type="image/png"
        )
        # Heatmap (XAI) Image
        firebase_xai_url = storage_manager.upload_file(
            result["xai_heatmap_path"], 
            remote_folder="hydration_images",
            local_url_prefix="/uploads",
            content_type="image/png"
        ) if result.get("xai_heatmap_path") else None

        # Update Result Object (Used by Frontend)
        if firebase_url:
            result["image_url"] = firebase_url
        if firebase_xai_url:
            result["xai_url"] = firebase_xai_url

        # Save to Database with new URL
        db_entry = LipAnalysis(
            user_id=current_user.id,
            image_path=firebase_url or result["saved_image_path"], # Store full URL/path
            prediction=result["prediction"],
            hydration_score=result["hydration_score"],
            confidence=result["confidence"]
        )
        db.add(db_entry)
        db.commit()

        # Fetch personalized suggestions from database
        prediction_context = {
            "lip_prediction": result["prediction"],
            "hydration_score": result["hydration_score"]
        }
        
        personalized_suggestions = fetch_personalized_suggestions(db, "lip", prediction_context)
        result["personalized_suggestions"] = personalized_suggestions

        return result
        
    except Exception as e:
        print(f"LIP PREDICT ERROR: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if 'temp_filename' in locals() and os.path.exists(temp_filename):
            os.remove(temp_filename)

# =====================================================
# ROUTES: HISTORY & DASHBOARD
# =====================================================

@router.get("/history/hydration")
def get_hydration_history(
    start_time: str = None,
    current_user: User = Depends(get_current_user), 
    db: Session = Depends(get_db)
):
    query = db.query(HydrationData).filter(HydrationData.user_id == current_user.id)
    
    if start_time:
        try:
            start_dt = datetime.fromisoformat(start_time.replace("Z", "+00:00"))
            start_dt = start_dt.astimezone(dt_module.timezone.utc).replace(tzinfo=None)
            query = query.filter(HydrationData.timestamp >= start_dt)
        except (ValueError, TypeError):
            pass

    entries = query.order_by(HydrationData.timestamp).all()
    return [{
        "date": e.timestamp.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "liters": e.recommended_liters,
        "risk": e.risk_level
    } for e in entries]

@router.get("/history/lip")
def get_lip_history(
    start_time: str = None,
    current_user: User = Depends(get_current_user), 
    db: Session = Depends(get_db)
):
    query = db.query(LipAnalysis).filter(LipAnalysis.user_id == current_user.id)
    
    if start_time:
        try:
            start_dt = datetime.fromisoformat(start_time.replace("Z", "+00:00"))
            start_dt = start_dt.astimezone(dt_module.timezone.utc).replace(tzinfo=None)
            query = query.filter(LipAnalysis.timestamp >= start_dt)
        except (ValueError, TypeError):
            pass

    entries = query.order_by(LipAnalysis.timestamp).all()
    return [{
        "date": e.timestamp.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "prediction": e.prediction,
        "hydration_score": e.hydration_score,
        "image_url": f"/uploads/{os.path.basename(e.image_path)}" if e.image_path else None
    } for e in entries]

@router.get("/history/lip-trends")
def get_lip_trends(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    try:
        month_ago = datetime.utcnow() - dt_module.timedelta(days=30)
        
        scans = db.query(LipAnalysis).filter(
            LipAnalysis.user_id == current_user.id,
            LipAnalysis.timestamp >= month_ago
        ).order_by(LipAnalysis.timestamp).all()
        
        if not scans:
            return {
                "trend_data": [],
                "summary": { "total_scans": 0, "avg_score": 0, "improvement": 0, "dehydrated_count": 0, "normal_count": 0 }
            }
        
        trend_data = [{
            "date": s.timestamp.strftime("%Y-%m-%d"),
            "score": s.hydration_score,
            "prediction": s.prediction,
            "confidence": s.confidence if hasattr(s, 'confidence') else None
        } for s in scans]
        
        scores = [s.hydration_score for s in scans]
        avg_score = sum(scores) / len(scores)
        
        if len(scans) >= 7:
            first_week_scores = scores[:min(7, len(scores))]
            last_week_scores = scores[-7:]
            improvement = (sum(last_week_scores) / len(last_week_scores)) - (sum(first_week_scores) / len(first_week_scores))
        else:
            improvement = 0
        
        dehydrated_count = sum(1 for s in scans if s.prediction == "Dehydrate")
        normal_count = len(scans) - dehydrated_count
        
        return {
            "trend_data": trend_data,
            "summary": {
                "total_scans": len(scans),
                "avg_score": round(avg_score, 1),
                "improvement": round(improvement, 1),
                "dehydrated_count": dehydrated_count,
                "normal_count": normal_count
            }
        }
    except Exception as e:
        print(f"LIP TRENDS ERROR: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/history/trends")
def get_hydration_trends(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    try:
        now_local = datetime.now()
        today_date_local = now_local.date()
        start_date_local = today_date_local - dt_module.timedelta(days=29)
        start_dt_query = datetime.utcnow() - dt_module.timedelta(days=32)
        
        entries = db.query(HydrationData).filter(
            HydrationData.user_id == current_user.id,
            HydrationData.timestamp >= start_dt_query
        ).all()
        
        daily_map = {}
        for i in range(30):
            d = start_date_local + dt_module.timedelta(days=i)
            daily_map[d.isoformat()] = 0.0

        hourly_map = {h: 0.0 for h in range(24)}
        today_iso = today_date_local.isoformat()
        
        for e in entries:
            local_dt = to_system_local(e.timestamp)
            entry_date = local_dt.date()
            if entry_date < start_date_local: continue
            
            date_str = entry_date.isoformat()
            
            slot = "Unknown"
            if e.input_data:
                slot = e.input_data.get("Time Slot (Select Your Current 4-Hour Window)", "Unknown")
            
            val = 0.0
            if e.input_data and "Water_Intake_Last_4_Hours" in e.input_data:
                try:
                    val = float(e.input_data["Water_Intake_Last_4_Hours"])
                except (TypeError, ValueError):
                    pass
            
            if date_str in daily_map:
                daily_map[date_str] += val
            
            if date_str == today_iso:
                hours = parse_slot_hours(slot)
                if not hours: hours = [local_dt.hour]
                
                per_hour_val = val / len(hours) if len(hours) > 0 else 0
                for h in hours:
                    if 0 <= h < 24: hourly_map[h] += per_hour_val
        
        full_data = [{"date": d, "liters": round(daily_map[d], 2)} for d in sorted(daily_map.keys())]
        hourly_data = [{"hour": f"{h:02d}:00", "liters": round(hourly_map[h], 2)} for h in range(24)]
        
        today_total = sum(hourly_map.values())
        weekly_data = full_data[-7:]
        weekly_total = sum(x["liters"] for x in weekly_data)
        monthly_total = sum(x["liters"] for x in full_data)
        
        return {
            "hourly": hourly_data,
            "weekly": weekly_data,
            "monthly": full_data,
            "today_total_liters": round(today_total, 2),
            "weekly_total_liters": round(weekly_total, 2),
            "monthly_total_liters": round(monthly_total, 2),
            "weekly_avg": round(weekly_total / 7, 2)
        }
    except Exception as e:
        print(f"TRENDS ERROR: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/tracker/dashboard")
def get_daily_dashboard(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    try:
        now_local = datetime.now()
        today_date_local = now_local.date()
        
        start_query = datetime.utcnow() - dt_module.timedelta(days=1, hours=6)
        
        recent_entries = db.query(HydrationData).filter(
            HydrationData.user_id == current_user.id,
            HydrationData.timestamp >= start_query
        ).all()
        
        total_intake = 0.0
        for entry in recent_entries:
            local_dt = to_system_local(entry.timestamp)
            if local_dt.date() != today_date_local: continue
                 
            if not entry.input_data: continue
            
            if "Water_Intake_Last_4_Hours" in entry.input_data:
                try: 
                    val = float(entry.input_data["Water_Intake_Last_4_Hours"])
                    total_intake += val
                except: continue
                    
        latest_hydration = db.query(HydrationData).filter(
            HydrationData.user_id == current_user.id
        ).order_by(HydrationData.timestamp.desc()).first()
        
        next_goal = latest_hydration.recommended_liters if latest_hydration else 0.0
        
        latest_lip = db.query(LipAnalysis).filter(
             LipAnalysis.user_id == current_user.id
        ).order_by(LipAnalysis.timestamp.desc()).first()
        
        lip_status = {
            "status": latest_lip.prediction if latest_lip else "Unknown",
            "score": latest_lip.hydration_score if latest_lip else 0,
            "last_updated": to_system_local(latest_lip.timestamp).strftime("%Y-%m-%dT%H:%M:%SZ") if latest_lip else None
        }
        
        weight = float(current_user.weight) if current_user.weight else 60.0
        height = float(current_user.height) if current_user.height else 170.0
        gender = str(current_user.gender) if current_user.gender else "Male"
        age = int(current_user.age) if current_user.age else 25
        
        base_goal = weight * 0.033
        height_add = 0.0
        if height > 185: height_add = 0.3
        elif height > 175: height_add = 0.1
        
        gender_add = 0.0
        if gender.lower() in ["male", "m", "man"]: gender_add = 0.5
        
        age_add = 0.0
        if age < 30: age_add = 0.2
        elif age > 55: age_add = -0.1

        weather_add = 0.0
        if latest_hydration and latest_hydration.input_data:
            try:
                last_temp = float(latest_hydration.input_data.get("Temperature_C", 25.0))
                if last_temp > 30:
                    weather_add = 0.5
                elif last_temp > 25:
                    weather_add = 0.2
            except (TypeError, ValueError):
                pass
        
        daily_goal = base_goal + height_add + gender_add + age_add + weather_add
        daily_goal = round(daily_goal, 2)
        
        percent = 0.0
        if daily_goal > 0:
            percent = (total_intake / daily_goal) * 100
            
        return {
            "date": today_date_local.isoformat(),
            "total_water_intake_today_liters": round(total_intake, 2),
            "next_4_hours_water_need_liters": round(next_goal, 2),
            "daily_goal_liters": daily_goal,
            "percentage_completed": round(percent, 1),
            "goal_status": "Goal Met!" if percent >= 100 else "Keep drinking water.",
            "current_lip_status": lip_status
        }
    except Exception as e:
        print(f"TRACKER ERROR: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/history/clear")
def clear_history(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    db.query(HydrationData).filter(HydrationData.user_id == current_user.id).delete()
    
    lip_entries = db.query(LipAnalysis).filter(LipAnalysis.user_id == current_user.id).all()
    for entry in lip_entries:
        if entry.image_path and os.path.exists(entry.image_path):
            try:
                os.remove(entry.image_path)
            except OSError:
                pass
    
    db.query(LipAnalysis).filter(LipAnalysis.user_id == current_user.id).delete()
    db.commit()
    return {"message": "History and Scan data cleared successfully"}
