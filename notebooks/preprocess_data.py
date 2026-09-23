import pandas as pd
import numpy as np
import re
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import json

# ─────────────────────────────────────
# 1. LOAD DATASET
# ─────────────────────────────────────

print("Loading dataset...")
df = pd.read_csv("../data/raw/maintenance_logs.csv")
print(f"Loaded {len(df)} records")
print(f"Columns: {list(df.columns)}")

# ─────────────────────────────────────
# 2. BASIC CHECKS
# ─────────────────────────────────────

print("\n--- BASIC CHECKS ---")

# Check missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Check duplicates
duplicates = df.duplicated().sum()
print(f"\nDuplicate Rows: {duplicates}")

# Check data types
print("\nData Types:")
print(df.dtypes)

# ─────────────────────────────────────
# 3. CLEAN MISSING VALUES
# ─────────────────────────────────────

print("\nCleaning missing values...")

# Drop rows with missing log entries
df = df.dropna(subset=["log_entry"])

# Fill missing technician with unknown
df["technician"] = df["technician"].fillna("Unknown")

print(f"Records after cleaning: {len(df)}")

# ─────────────────────────────────────
# 4. CLEAN TEXT
# ─────────────────────────────────────

def clean_text(text):
    """
    Clean maintenance log text
    Step by step cleaning
    """
    # Step 1: Convert to lowercase
    text = text.lower()
    
    # Step 2: Remove special characters
    # Keep only letters, numbers, spaces
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
    
    # Step 3: Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Step 4: Remove leading/trailing spaces
    text = text.strip()
    
    return text

print("\nCleaning text...")
df["clean_log"] = df["log_entry"].apply(clean_text)

# Show before and after example
print("\n--- TEXT CLEANING EXAMPLE ---")
print(f"Before: {df['log_entry'].iloc[0]}")
print(f"After : {df['clean_log'].iloc[0]}")

# ─────────────────────────────────────
# 5. ADD USEFUL FEATURES
# ─────────────────────────────────────

print("\nAdding features...")

# Feature 1: Log length (number of words)
df["log_length"] = df["clean_log"].apply(
    lambda x: len(x.split())
)

# Feature 2: Has urgent keywords
urgent_words = [
    "urgent", "emergency", "critical", "immediate",
    "seized", "stopped", "halted", "failure", "broken",
    "collapsed", "snapped", "cracked", "spark", "burning"
]

def has_urgent_keyword(text):
    text_lower = text.lower()
    for word in urgent_words:
        if word in text_lower:
            return 1
    return 0

df["has_urgent"] = df["log_entry"].apply(has_urgent_keyword)

# Feature 3: Is it a failure log
df["is_failure"] = df["failure_type"].apply(
    lambda x: 0 if x in ["No Failure", "Pre-Failure Warning"] else 1
)

# Feature 4: Convert date to datetime
df["date"] = pd.to_datetime(df["date"])
df["month"] = df["date"].dt.month
df["day_of_week"] = df["date"].dt.dayofweek

print("Features added successfully!")

# ─────────────────────────────────────
# 6. ENCODE LABELS
# ─────────────────────────────────────

print("\nEncoding labels...")

# Encode failure_type (main label)
failure_encoder = LabelEncoder()
df["failure_label"] = failure_encoder.fit_transform(
    df["failure_type"]
)

# Encode severity
severity_encoder = LabelEncoder()
df["severity_label"] = severity_encoder.fit_transform(
    df["severity"]
)

# Save label mappings
failure_mapping = {
    int(i): label 
    for i, label in enumerate(failure_encoder.classes_)
}

severity_mapping = {
    int(i): label 
    for i, label in enumerate(severity_encoder.classes_)
}

print("\nFailure Type Mapping:")
for k, v in failure_mapping.items():
    print(f"  {k} → {v}")

print("\nSeverity Mapping:")
for k, v in severity_mapping.items():
    print(f"  {k} → {v}")

# Save mappings to JSON
os.makedirs("../data/clean", exist_ok=True)

with open("../data/clean/failure_mapping.json", "w") as f:
    json.dump(failure_mapping, f, indent=2)

with open("../data/clean/severity_mapping.json", "w") as f:
    json.dump(severity_mapping, f, indent=2)

print("\nLabel mappings saved!")

# ─────────────────────────────────────
# 7. SPLIT INTO TRAIN / VAL / TEST
# ─────────────────────────────────────

print("\nSplitting dataset...")

# First split: 80% train, 20% temp
train_df, temp_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42,
    stratify=df["failure_label"]
)

# Second split: 10% val, 10% test
val_df, test_df = train_test_split(
    temp_df,
    test_size=0.5,
    random_state=42,
    stratify=temp_df["failure_label"]
)

print(f"Train set : {len(train_df)} records (80%)")
print(f"Val set   : {len(val_df)} records (10%)")
print(f"Test set  : {len(test_df)} records (10%)")

# ─────────────────────────────────────
# 8. SAVE CLEANED DATA
# ─────────────────────────────────────

print("\nSaving cleaned data...")

# Save full cleaned dataset
df.to_csv("../data/clean/maintenance_logs_clean.csv",
          index=False)

# Save train/val/test splits
train_df.to_csv("../data/clean/train.csv", index=False)
val_df.to_csv("../data/clean/val.csv", index=False)
test_df.to_csv("../data/clean/test.csv", index=False)

print("All files saved!")

# ─────────────────────────────────────
# 9. FINAL SUMMARY
# ─────────────────────────────────────

print("\n" + "="*50)
print("PREPROCESSING COMPLETE")
print("="*50)
print(f"Total Records    : {len(df)}")
print(f"Train Records    : {len(train_df)}")
print(f"Val Records      : {len(val_df)}")
print(f"Test Records     : {len(test_df)}")
print(f"Features Created : log_length, has_urgent,")
print(f"                   is_failure, month, day_of_week")
print(f"\nFiles Saved:")
print(f"  data/clean/maintenance_logs_clean.csv")
print(f"  data/clean/train.csv")
print(f"  data/clean/val.csv")
print(f"  data/clean/test.csv")
print(f"  data/clean/failure_mapping.json")
print(f"  data/clean/severity_mapping.json")

# Show sample cleaned record
print("\n--- SAMPLE CLEANED RECORD ---")
sample = df.iloc[0]
print(f"Date         : {sample['date']}")
print(f"Machine      : {sample['machine_id']}")
print(f"Raw Log      : {sample['log_entry']}")
print(f"Clean Log    : {sample['clean_log']}")
print(f"Failure Type : {sample['failure_type']}")
print(f"Label        : {sample['failure_label']}")
print(f"Severity     : {sample['severity']}")
print(f"Has Urgent   : {sample['has_urgent']}")
print(f"Log Length   : {sample['log_length']} words")