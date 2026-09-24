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
# 5. COPY APPLICATION CODE & DATA
# ─────────────────────────────────────
COPY api/ ./api/
COPY models/ ./models/
# Copy the entire data directory so all files (including maintenance_logs_clean.csv) are included
COPY data/ ./data/

# ─────────────────────────────────────
# 6. EXPOSE PORT AND RUN
# ─────────────────────────────────────
EXPOSE 8000

CMD ["python", "-m", "api.main"]