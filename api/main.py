from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import torch
import json
import re
from typing import List, Optional
from transformers import DistilBertTokenizer

from models.bert_classifier.model import BERTFailureClassifier
from models.autoencoder.model import LogAutoencoder
from models.autoencoder.dataset import get_bert_embedding
from models.lstm_predictor.predict import predict_risk, predict_from_logs

# ─────────────────────────────────────
# 1. CONFIGURATION
# ─────────────────────────────────────

CONFIG = {
    'bert_model_path'        : 'models/bert_classifier/best_model.pt',
    'failure_mapping_path'   : 'data/clean/failure_mapping.json',
    'autoencoder_model_path' : 'models/autoencoder/saved/best_model.pt',
    'vectorizer_path'        : 'models/autoencoder/saved/tfidf_vectorizer.pkl',
    'max_length'             : 128,
    'num_classes'            : 6,
    'encoding_dim'           : 64,
    'input_dim'              : 768,
    'dropout'                : 0.2,
    'threshold_path'         : 'models/autoencoder/saved/threshold.json',
}


# ─────────────────────────────────────
# 2. DETECT DEVICE (M2 Mac = mps)
# ─────────────────────────────────────

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print(f"API using device: {device} ✅")


# ─────────────────────────────────────
# 3. LOAD BERT CLASSIFIER (once at startup)
# ─────────────────────────────────────

print("\nLoading BERT classifier...")

bert_tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

bert_model = BERTFailureClassifier(num_classes=CONFIG['num_classes'])
bert_model.load_state_dict(
    torch.load(CONFIG['bert_model_path'], map_location=device)
)
bert_model.to(device)
bert_model.eval()

with open(CONFIG['failure_mapping_path']) as f:
    failure_mapping = json.load(f)

print("BERT loaded ✅")


# ─────────────────────────────────────
# 4b. LOAD AUTOENCODER (once at startup)
# ─────────────────────────────────────

autoencoder = LogAutoencoder(
    input_dim=CONFIG['input_dim'],
    encoding_dim=CONFIG['encoding_dim'],
    dropout=CONFIG['dropout']
).to(device)
autoencoder.load_state_dict(
    torch.load(CONFIG['autoencoder_model_path'], map_location=device)
)
autoencoder.eval()

print("Autoencoder loaded ✅")

# ─────────────────────────────────────
# 4b. LOAD ANOMALY THRESHOLD (from saved JSON, not hardcoded)
# ─────────────────────────────────────

print("\nLoading anomaly threshold...")

with open(CONFIG['threshold_path']) as f:
    threshold_data = json.load(f)

ANOMALY_THRESHOLD = threshold_data['anomaly_threshold']
THRESHOLD_PERCENTILE = threshold_data['percentile']

print(f"Threshold loaded: {ANOMALY_THRESHOLD:.6f} ({THRESHOLD_PERCENTILE}th percentile) ✅")

BORDERLINE_MULTIPLIER = 1.5
BORDERLINE_THRESHOLD = ANOMALY_THRESHOLD * BORDERLINE_MULTIPLIER
print(f"Borderline zone: {ANOMALY_THRESHOLD:.6f} to {BORDERLINE_THRESHOLD:.6f} ✅")


# ─────────────────────────────────────
# 5. LSTM RISK PREDICTOR
# ─────────────────────────────────────

print("\nLSTM risk predictor ready ✅")


# ─────────────────────────────────────
# 6. REQUEST/RESPONSE SCHEMAS
# ─────────────────────────────────────

class LogRequest(BaseModel):
    """Expected input: a single maintenance log entry (supports log_text or clean_log)"""
    log_text: Optional[str] = ""
    clean_log: Optional[str] = ""


class PredictionResponse(BaseModel):
    """Combined output from BERT + autoencoder + rule-based insights"""
    predicted_failure_type: str
    confidence: float
    is_anomaly: bool
    anomaly_status: str
    reconstruction_error: float
    anomaly_threshold: float
    component: str
    severity: str
    risk_level: str
    urgency: str


class MachineRequest(BaseModel):
    """Expected input: a machine ID to check risk for"""
    machine_id: str


class RiskResponse(BaseModel):
    """Output from LSTM risk prediction"""
    machine_id: str
    risk_probability: float
    is_at_risk: bool


# ─────────────────────────────────────
# 7. BERT PREDICTION FUNCTION
# ─────────────────────────────────────

def predict_failure_type(log_text: str):
    """Runs BERT on a single log and returns failure type + confidence"""

    encoding = bert_tokenizer(
        log_text,
        max_length=CONFIG['max_length'],
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )

    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    with torch.no_grad():
        outputs = bert_model(input_ids, attention_mask)
        probs = torch.softmax(outputs, dim=1)
        pred = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred].item()

    failure_type = failure_mapping[str(pred)]

    return failure_type, confidence


# ─────────────────────────────────────
# 8. AUTOENCODER SCORING FUNCTION
# ─────────────────────────────────────

def get_anomaly_score(log_text: str):
    embedding = get_bert_embedding(log_text)
    embedding_tensor = torch.tensor(embedding, dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        reconstructed = autoencoder(embedding_tensor)
        error = torch.mean((reconstructed - embedding_tensor) ** 2).item()

    if error > BORDERLINE_THRESHOLD:
        anomaly_status = "anomaly"
    elif error > ANOMALY_THRESHOLD:
        anomaly_status = "borderline"
    else:
        anomaly_status = "normal"

    is_anomaly = anomaly_status == "anomaly"

    return error, is_anomaly, anomaly_status


# ─────────────────────────────────────
# 8b. DERIVED INSIGHTS & VALIDATION
# ─────────────────────────────────────

COMPONENT_KEYWORDS = {
    'bearing'    : 'Bearing',
    'gearbox'    : 'Gearbox',
    'gear'       : 'Gearbox',
    'motor'      : 'Motor',
    'shaft'      : 'Shaft',
    'pump'       : 'Pump',
    'coupling'   : 'Coupling',
    'valve'      : 'Valve',
    'sensor'     : 'Sensor',
    'electrical' : 'Electrical Panel',
    'panel'      : 'Electrical Panel',
    'conveyor'   : 'Conveyor',
}

HIGH_SEVERITY_KEYWORDS = ['spark', 'fire', 'smoke', 'broken', 'leak', 'overheat', 'excessive', 'seized', 'cracked']
MEDIUM_SEVERITY_KEYWORDS = ['wear', 'noise', 'vibration', 'minor', 'monitor', 'elevated', 'drift', 'play', 'degraded']

MAINTENANCE_KEYWORDS = [
    'inspection', 'maintenance', 'check', 'sensor', 'pump', 'bearing', 'valve', 
    'gearbox', 'motor', 'shaft', 'coupling', 'panel', 'conveyor', 'machine', 
    'hydraulic', 'belt', 'vibration', 'noise', 'temperature', 'pressure', 'leak', 
    'wear', 'operating', 'normal', 'status', 'reading', 'service', 'oil', 'voltage',
    'fault', 'warning', 'failure', 'routine', 'shift', 'technician', 'report',
    'observed', 'noted', 'abnormal', 'stable', 'performance', 'range', 'limits',
    'gear', 'fluid', 'rpm', 'current', 'spark', 'smoke', 'plc'
]

def is_valid_maintenance_log(text: str) -> bool:
    """Validates if input text resembles a maintenance log"""
    text_lower = text.lower()
    matches = sum(1 for kw in MAINTENANCE_KEYWORDS if kw in text_lower)
    return matches >= 1


def detect_component(log_text: str) -> str:
    text_lower = log_text.lower()
    for keyword, component in COMPONENT_KEYWORDS.items():
        if keyword in text_lower:
            return component
    return "Unspecified"


def detect_severity(log_text: str) -> str:
    text_lower = log_text.lower()
    for word in HIGH_SEVERITY_KEYWORDS:
        if word in text_lower:
            if f"no {word}" in text_lower or f"no {word}s" in text_lower or f"without {word}" in text_lower:
                continue
            return "High"
            
    for word in MEDIUM_SEVERITY_KEYWORDS:
        if word in text_lower:
            if f"no {word}" in text_lower or f"no {word}s" in text_lower or f"without {word}" in text_lower:
                continue
            return "Medium"
            
    return "Low"


def compute_risk_level(failure_type: str, severity: str, anomaly_status: str, confidence: float):
    if failure_type == "No Failure":
        if anomaly_status == "anomaly" and confidence > 0.7:
            return "🟡 Moderate", "Review Recommended"
        if anomaly_status == "borderline":
            return "🟡 Borderline", "Manual Review Suggested"
        return "🟢 Normal", "No Immediate Action Required"

    if severity == "High" and anomaly_status == "anomaly":
        return "🔴 Critical", "Immediate Inspection"
    if severity == "Medium" and anomaly_status == "anomaly":
        return "🟠 Warning", "Schedule Inspection Soon"
    if anomaly_status == "borderline":
        return "🟡 Borderline", "Manual Review Suggested"
    if anomaly_status == "normal" and severity == "Low":
        return "🟢 Normal", "No Action Needed"
    return "🟡 Moderate", "Monitor Closely"


# ─────────────────────────────────────
# 9. INITIALIZE FASTAPI APP
# ─────────────────────────────────────

app = FastAPI(
    title="Industrial Equipment Failure Prediction API",
    description="Predicts failure type, flags anomalies, and estimates machine risk from maintenance logs",
    version="1.0.0"
)


# ─────────────────────────────────────
# 10. API ENDPOINTS
# ─────────────────────────────────────

@app.get("/")
def read_root():
    return {"status": "API is running ✅"}


@app.post("/predict", response_model=PredictionResponse)
def predict(request: LogRequest):
    log_text_val = (request.log_text or "").strip()
    clean_log_val = (request.clean_log or "").strip()
    target_log = log_text_val if log_text_val else clean_log_val

    if not target_log:
        raise HTTPException(status_code=400, detail="log_text cannot be empty")

    if not is_valid_maintenance_log(target_log):
        raise HTTPException(status_code=400, detail="Please enter a valid maintenance log")

    failure_type, confidence = predict_failure_type(target_log)
    error, is_anomaly, anomaly_status = get_anomaly_score(target_log)

    component = detect_component(target_log)
    severity = detect_severity(target_log)
    risk_level, urgency = compute_risk_level(failure_type, severity, anomaly_status, confidence)

    return PredictionResponse(
        predicted_failure_type=failure_type,
        confidence=round(confidence, 4),
        is_anomaly=is_anomaly,
        anomaly_status=anomaly_status,
        reconstruction_error=round(error, 6),
        anomaly_threshold=ANOMALY_THRESHOLD,
        component=component,
        severity=severity,
        risk_level=risk_level,
        urgency=urgency
    )


@app.post("/predict_risk", response_model=RiskResponse)
def predict_machine_risk(request: MachineRequest):
    if not request.machine_id.strip():
        raise HTTPException(status_code=400, detail="machine_id cannot be empty")

    result = predict_risk(request.machine_id)

    if 'error' in result:
        raise HTTPException(status_code=400, detail=result['error'])

    return RiskResponse(
        machine_id=result['machine_id'],
        risk_probability=result['risk_probability'],
        is_at_risk=result['is_at_risk']
    )


# ─────────────────────────────────────
# 10b. LSTM SEQUENCE ENDPOINT
# ─────────────────────────────────────

class SequenceRequest(BaseModel):
    logs: List[str]


class SequenceResponse(BaseModel):
    risk_probability: float
    is_at_risk: bool
    threshold: float
    logs_used: int


def clean_log_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    return re.sub(r'\s+', ' ', text).strip()


@app.post("/predict_sequence", response_model=SequenceResponse)
def predict_sequence_risk(request: SequenceRequest):
    logs = [clean_log_text(t) for t in request.logs if t.strip()]

    result = predict_from_logs(logs)

    if 'error' in result:
        raise HTTPException(status_code=400, detail=result['error'])

    warning_words = ['degraded', 'leak', 'play', 'misalignment', 'elevated', 'noise', 'looser', 'drift', 'minor', 'warning', 'fault', 'complete oil change']
    warning_count = sum(1 for log in logs if any(w in log for w in warning_words))
    
    current_risk = result['risk_probability']
    if warning_count >= 3 and current_risk < 0.45:
        boosted_risk = min(0.55 + (warning_count * 0.03), 0.80)
        result['risk_probability'] = boosted_risk
        result['is_at_risk'] = boosted_risk >= result.get('threshold', 0.5)

    return SequenceResponse(
        risk_probability=result['risk_probability'],
        is_at_risk=result['is_at_risk'],
        threshold=result['threshold'],
        logs_used=len(result['logs_used'])
    )


# ─────────────────────────────────────
# 11. RUN SERVER (for local testing)
# ─────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)