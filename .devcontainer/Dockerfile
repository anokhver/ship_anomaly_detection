# 1. Backend and frontend together
FROM python:3.11-slim

# Node.js for Vue + system deps
RUN apt-get update && \
    apt-get install -y \
        curl \
        gnupg \
        git \
        gcc \
        libpq-dev \
        python3-dev \
        build-essential \
        tree \
    && curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y nodejs \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Set work directory
WORKDIR /workspace

# Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt


# Install frontend dependencies (if not using multi-stage build)
COPY frontend/package*.json frontend/
RUN cd frontend && npm install



# Optional: build frontend (Vue) on container start
# You can also leave this to a manual npm run build
# RUN cd frontend && npm run build

# Expose both ports
EXPOSE 8000 5173

# run the backend server
## python manage.py runserver 0.0.0.0:8000