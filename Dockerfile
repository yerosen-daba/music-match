FROM python:3.11-slim

# Install system dependencies required for librosa audio processing
RUN apt-get update && apt-get install -y \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Run the FastAPI server
CMD ["uvicorn", "app_v2:app", "--host", "0.0.0.0", "--port", "10000"]
