# Serves the production churn model from models/ through the FastAPI app.
FROM python:3.12-slim

# LightGBM needs OpenMP at runtime
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements-serve.txt .
RUN pip install --no-cache-dir -r requirements-serve.txt

COPY clientpulse/ clientpulse/
COPY models/ models/

ENV CLIENTPULSE_MLFLOW=0
RUN useradd --create-home app && chown -R app /app
USER app

EXPOSE 8000
CMD ["sh", "-c", "uvicorn clientpulse.api:app --host 0.0.0.0 --port ${PORT:-8000}"]
