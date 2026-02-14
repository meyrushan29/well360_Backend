
# ------------------------------------------------------------
# DOCKERFILE FOR WELL360 BACKEND (Render / Railway / AWS)
# ------------------------------------------------------------

# 1. Base Image: Python 3.10 is stable and good for ML
FROM python:3.10-slim

# 2. Prevent Python from buffering stdout/stderr (see logs immediately)
ENV PYTHONUNBUFFERED=1

# 3. Set Working Directory
WORKDIR /app

# 4. Install System Dependencies (OpenCV needs libgl1)
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 5. Copy Dependencies First (Layer Cache Optimization)
COPY requirements.txt .

# 6. Install Python Dependencies
# Upgrade pip first to avoid issues
RUN pip install --upgrade pip
# Install requirements with no cache to save space in image
RUN pip install --no-cache-dir -r requirements.txt
# Install Gunicorn strictly for production server
RUN pip install gunicorn

# 7. Copy Application Code
COPY . .

# 8. Create Directories for Persistence (Data & Uploads)
# Render/Railway will mount Volumes here.
RUN mkdir -p /app/hydration
RUN mkdir -p /app/img/uploads
RUN mkdir -p /app/img/fitness_processed
RUN mkdir -p /app/temp

# 9. Expose Port (Standard convention, but platforms set PORT env var)
EXPOSE 8000

# 10. Start Command
# Using gunicorn with Uvicorn worker for production performance
# Reads PORT environment variable, defaults to 8000
CMD exec gunicorn main:app --workers 1 --worker-class uvicorn.workers.UvicornWorker --bind 0.0.0.0:$PORT
