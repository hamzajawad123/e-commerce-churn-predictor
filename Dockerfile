# API + Streamlit image; Feast online store is baked in at build time
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/
COPY src/ ./src/
COPY feature_repo/ ./feature_repo/

EXPOSE 8000 8501

# Compose overrides CMD for streamlit; one ENTRYPOINT covers both services
ENTRYPOINT ["python", "-m"]
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
