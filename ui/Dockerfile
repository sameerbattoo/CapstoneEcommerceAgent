FROM python:3.13-slim

WORKDIR /app

# Install system dependencies for psycopg2 and other packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better layer caching
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Copy the UI application
COPY . .

# Expose Streamlit default port
EXPOSE 8501

# Health check for ECS
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Run Streamlit
ENTRYPOINT ["streamlit", "run", "orch_web_app_cognito.py", "--server.port=8501", "--server.address=0.0.0.0"]
