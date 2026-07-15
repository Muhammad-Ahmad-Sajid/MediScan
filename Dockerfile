# Base image with glibc for PyTorch and OpenCV compatibility
FROM python:3.10-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Set working directory
WORKDIR /app

# Install system dependencies for OpenCV, PostgreSQL, and health checks
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq-dev \
    gcc \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install python-jose[cryptography] python-multipart email-validator reportlab "bcrypt<4.0.0"
RUN apt-get update && apt-get install -y libgl1 && rm -rf /var/lib/apt/lists/*

# Copy the entire project
COPY . .

# Create required directories
RUN mkdir -p uploads heatmaps reports checkpoints mlruns

# Expose FastAPI port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Start FastAPI application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
