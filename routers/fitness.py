from fastapi import APIRouter, Depends, HTTPException, File, UploadFile, Form
import os
import uuid
from core.models import User
from core.deps import get_current_user

router = APIRouter(
    prefix="/predict/fitness",
    tags=["Fitness"]
)

@router.post("/video")
def predict_fitness_video(
    video: UploadFile = File(...),
    enable_heatmap: bool = Form(True),
    current_user: User = Depends(get_current_user)
):
    try:
        try:
            from fitness.api_handler import get_processor
        except Exception as imp_err:
            raise HTTPException(
                status_code=503,
                detail=f"Fitness processor unavailable: {imp_err}"
            )

        os.makedirs("temp", exist_ok=True)
        temp_filename = f"temp/{uuid.uuid4()}_{video.filename}"
        
        with open(temp_filename, "wb") as f:
            f.write(video.file.read())
            
        output_dir = "img/fitness_processed"
        processor = get_processor()
        result = processor.process_video(temp_filename, output_dir=output_dir, enable_heatmap=enable_heatmap)
        
        if "error" in result:
             if "No human detected" in result["error"]:
                 raise HTTPException(status_code=400, detail=result["error"])
             raise HTTPException(status_code=500, detail=result["error"])
        
        result["video_url"] = f"/fitness_videos/{result['processed_video_filename']}"
        
        if "video_filename_normal" in result:
             result["video_url_normal"] = f"/fitness_videos/{result['video_filename_normal']}"
        if "video_filename_heatmap" in result:
             result["video_url_heatmap"] = f"/fitness_videos/{result['video_filename_heatmap']}"
        
        return result
        
    except Exception as e:
        print(f"FITNESS PREDICT ERROR: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if 'temp_filename' in locals() and os.path.exists(temp_filename):
            os.remove(temp_filename)
