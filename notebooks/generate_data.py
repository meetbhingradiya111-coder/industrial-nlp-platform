import pandas as pd
import random
from datetime import datetime, timedelta

# ─────────────────────────────────────
# 1. DEFINE LOG TEMPLATES
# ─────────────────────────────────────

# Normal logs (no failure)
normal_logs = [
    "Routine inspection completed. All systems normal.",
    "Regular maintenance done. No issues found.",
    "Lubrication applied. Machine running smoothly.",
    "Checked all components. Everything in good condition.",
    "Scheduled maintenance completed successfully.",
    "Visual inspection done. No abnormalities detected.",
    "Oil levels checked and topped up. Normal operation.",
    "Belt tension checked. Within normal range.",
    "Cooling system inspected. Temperature normal.",
    "Electrical connections checked. All secure.",
]

# Early warning logs (pre-failure)
warning_logs = [
    "Minor vibration noticed during operation.",
    "Slight noise from motor shaft. Monitoring required.",
    "Temperature slightly higher than normal. Noted.",
    "Small oil leak noticed near base. Cleaned.",
    "Belt showing minor wear. Will monitor.",
    "Unusual sound from bearing area. Lubrication applied.",
    "Motor running slightly hot. Ventilation checked.",
    "Minor misalignment noticed. Adjusted slightly.",
    "Vibration increasing compared to last check.",
    "Intermittent noise from gearbox. Under observation.",
]

# Bearing failure logs
bearing_logs = [
    "Loud grinding noise from motor shaft bearing.",
    "Bearing worn out completely. Replacement needed urgently.",
    "Severe vibration due to bearing failure detected.",
    "Bearing seized. Machine stopped immediately.",
    "Metal shavings found near bearing housing.",
    "Bearing temperature critically high. Emergency stop.",
    "Grinding and squealing noise from bearing assembly.",
    "Bearing clearance exceeded limit. Immediate replacement needed.",
    "Shaft bearing collapsed. Production halted.",
    "Excessive bearing wear detected during inspection.",
]

# Electrical fault logs
electrical_logs = [
    "Motor overheating. Electrical fault suspected.",
    "Sparks observed from motor winding.",
    "Burning smell from electrical panel. Investigated.",
    "Circuit breaker tripped multiple times today.",
    "Voltage fluctuation causing motor instability.",
    "Insulation breakdown detected in motor winding.",
    "Electrical short circuit in control panel.",
    "Motor drawing excessive current. Shutdown initiated.",
    "Wiring damaged near junction box. Replaced.",
    "Control panel showing error codes. Electrician called.",
]

# Lubrication failure logs
lubrication_logs = [
    "Severe oil leak from main shaft seal.",
    "Lubrication system blocked. Oil not reaching bearing.",
    "Oil contamination detected. System flushed.",
    "Grease dried out completely in bearing housing.",
    "Oil pump failure causing lubrication system breakdown.",
    "Insufficient lubrication causing excessive wear.",
    "Oil level critically low. Machine at risk.",
    "Lubrication line broken. Emergency repair done.",
    "Contaminated oil causing corrosion in components.",
    "Oil viscosity degraded. Complete oil change needed.",
]

# Mechanical fault logs
mechanical_logs = [
    "Shaft misalignment causing excessive vibration.",
    "Coupling broken between motor and gearbox.",
    "Gear teeth worn out. Replacement scheduled.",
    "Cracked housing detected near drive end.",
    "Belt snapped during operation. Production stopped.",
    "Pulley damaged causing belt slippage.",
    "Structural crack found in machine frame.",
    "Bolt loosening causing machine instability.",
    "Gearbox teeth stripped. Immediate replacement needed.",
    "Machine base cracked due to excessive vibration.",
]

# ─────────────────────────────────────
# 2. DEFINE MACHINES AND COMPONENTS
# ─────────────────────────────────────

machines = [
    "M-101", "M-102", "M-103", "M-104", "M-105",
    "M-201", "M-202", "M-203", "M-204", "M-205",
    "M-301", "M-302", "M-303", "M-304", "M-305",
]

technicians = [
    "Rajesh Kumar", "Suresh Patel", "Amit Shah",
    "Vikram Singh", "Deepak Mehta", "Ravi Verma",
    "Sunil Joshi", "Mahesh Tiwari"
]

# ─────────────────────────────────────
# 3. DEFINE FAILURE CATEGORIES
# ─────────────────────────────────────

failure_categories = {
    "No Failure"          : (normal_logs,       "None",    "None"),
    "Pre-Failure Warning" : (warning_logs,       "None",    "Low"),
    "Bearing Failure"     : (bearing_logs,       "Bearing", "High"),
    "Electrical Fault"    : (electrical_logs,    "Motor",   "High"),
    "Lubrication Failure" : (lubrication_logs,   "Shaft",   "Medium"),
    "Mechanical Fault"    : (mechanical_logs,    "Gearbox", "High"),
}

# ─────────────────────────────────────
# 4. GENERATE DATASET
# ─────────────────────────────────────

def generate_dataset(num_records=5000):
    records = []
    start_date = datetime(2024, 1, 1)

    for i in range(num_records):
        # Pick random machine and technician
        machine_id  = random.choice(machines)
        technician  = random.choice(technicians)

        # Pick random date
        random_days = random.randint(0, 365)
        date        = start_date + timedelta(days=random_days)

        # Pick failure category
        # More normal logs than failure logs (realistic)
        weights = [40, 20, 10, 10, 10, 10]
        category = random.choices(
            list(failure_categories.keys()),
            weights=weights
        )[0]

        # Get log template and labels
        logs, component, severity = failure_categories[category]
        log_entry = random.choice(logs)

        # Add some randomness to logs
        prefixes = [
            "", "During shift inspection: ", 
            "Technician report: ", "Maintenance log: ",
            "Shift handover note: "
        ]
        log_entry = random.choice(prefixes) + log_entry

        records.append({
            "date"        : date.strftime("%Y-%m-%d"),
            "machine_id"  : machine_id,
            "technician"  : technician,
            "log_entry"   : log_entry,
            "failure_type": category,
            "component"   : component,
            "severity"    : severity,
        })

    return pd.DataFrame(records)


# ─────────────────────────────────────
# 5. SAVE DATASET
# ─────────────────────────────────────

print("Generating dataset...")
df = generate_dataset(num_records=5000)

# Sort by date
df = df.sort_values("date").reset_index(drop=True)

# Save to CSV
df.to_csv("../data/raw/maintenance_logs.csv", index=False)
print(f"Dataset saved! Total records: {len(df)}")

# ─────────────────────────────────────
# 6. EXPLORE DATASET
# ─────────────────────────────────────

print("\n--- DATASET OVERVIEW ---")
print(f"Total Records  : {len(df)}")
print(f"Total Machines : {df['machine_id'].nunique()}")
print(f"Date Range     : {df['date'].min()} to {df['date'].max()}")

print("\n--- FAILURE TYPE DISTRIBUTION ---")
print(df['failure_type'].value_counts())

print("\n--- SEVERITY DISTRIBUTION ---")
print(df['severity'].value_counts())

print("\n--- SAMPLE RECORDS ---")
print(df.head(10).to_string())