FROM python:3.11-slim

WORKDIR /app

# Install deps first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code (main.py, openclaw.json, SKILL.md, etc.)
COPY . .

# Non-root user for security
RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

# Shell form so $PORT is expanded at runtime (Render sets PORT dynamically)
CMD uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}
