# One image for every Python service (API, training job and MLflow server).
FROM python:3.12-slim

WORKDIR /app

# Install the pinned dependencies first, so this slow layer is cached between builds.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Then install our own package.
COPY pyproject.toml .
COPY src ./src
RUN pip install --no-cache-dir --no-deps .

# Do not run as root: create a normal user that owns the app and the MLflow data folder.
RUN useradd --create-home appuser \
    && mkdir -p /app/logs /app/metrics /mlflow \
    && chown -R appuser:appuser /app /mlflow
USER appuser

EXPOSE 8000
CMD ["uvicorn", "insurance.api:app", "--host", "0.0.0.0", "--port", "8000"]
