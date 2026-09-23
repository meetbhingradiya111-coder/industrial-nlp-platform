# ─────────────────────────────────────
# 1. BASE IMAGE
# ─────────────────────────────────────
FROM python:3.9-slim

# ─────────────────────────────────────
# 2. SYSTEM DEPENDENCIES
# ─────────────────────────────────────
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# ─────────────────────────────────────
# 3. WORKING DIRECTORY
# ─────────────────────────────────────
WORKDIR /app

# ─────────────────────────────────────
# 4. INSTALL PYTHON DEPENDENCIES
# ─────────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ─────────────────────────────────────
# 5. COPY APPLICATION CODE
# ─────────────────────────────────────
COPY api/ ./api/
COPY models/ ./models/
COPY data/clean/failure_mapping.json ./data/clean/failure_mapping.json
COPY data/clean/severity_mapping.json ./data/clean/severity_mapping.json

# ─────────────────────────────────────
# 6. EXPOSE PORT AND RUN
# ─────────────────────────────────────
EXPOSE 8000

CMD ["python", "-m", "api.main"]