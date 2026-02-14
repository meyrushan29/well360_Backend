from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import os

from core.database import engine, Base

# Import Routers
from routers import auth, hydration, fitness, mental_health, hydration_admin

# =====================================================
# APP INITIALIZATION & DB SETUP
# =====================================================

# Create Tables
Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="Well360 Backend API",
    version="2.1.0",
    description="Unified API for Hydration, Fitness, and Mental Health modules."
)

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    import traceback
    traceback.print_exc()
    return JSONResponse(
        status_code=500,
        content={"detail": str(exc)},
    )

# CORS SETUP (allow Flutter web and any origin)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

# === DEBUG: GLOBAL REQUEST LOGGING ===
import time
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    print(f"\n[REQUEST] {request.method} {request.url}")
    try:
        response = await call_next(request)
        process_time = time.time() - start_time
        print(f"[RESPONSE] {request.method} {request.url} -> {response.status_code} ({process_time:.2f}s)")
        return response
    except Exception as e:
        print(f"[ERROR] Request Failed: {e}")
        raise e

# =====================================================
# ROUTER REGISTRATION
# =====================================================
app.include_router(auth.router)
app.include_router(hydration.router)
app.include_router(hydration_admin.router)  # Admin endpoints for managing suggestions
app.include_router(fitness.router)
app.include_router(mental_health.router)

@app.get("/api-status")
def api_status():
    return {
        "status": "Well360 API Running", 
        "modules": ["Hydration", "Fitness", "MentalHealth"]
    }

# Root endpoint for health checks (prevents 404 on /)
@app.get("/")
def root():
    return {
        "message": "Well360 Backend is Live",
        "docs_url": "/docs",
        "redoc_url": "/redoc"
    }

# =====================================================
# STATIC FILES & UPLOADS
# =====================================================

# Mount user uploads (SAFE persistence, outside static build folder)
os.makedirs("img/uploads", exist_ok=True)
app.mount("/uploads", StaticFiles(directory="img/uploads"), name="uploads")

# Mount fitness processed videos
os.makedirs("img/fitness_processed", exist_ok=True)
app.mount("/fitness_videos", StaticFiles(directory="img/fitness_processed"), name="fitness_videos")


