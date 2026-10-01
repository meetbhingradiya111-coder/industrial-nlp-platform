import pandas as pd
import numpy as np

# ─────────────────────────────────────
# 1. CONFIGURATION
# ─────────────────────────────────────

CONFIG = {
    'train_path'   : 'data/sequences/train.csv',
    'val_path'     : 'data/sequences/val.csv',
        'horizon'      : 5,
    'sequence_len' : 10,
    'failure_types': [
        'Bearing Failure',
        'Electrical Fault',
        'Lubrication Failure',
        'Mechanical Fault',
    ],
    'warning_type' : 'Pre-Failure Warning',
}


# ─────────────────────────────────────
# 2. LOAD DATA
# ─────────────────────────────────────

def load_split(path):
    df = pd.read_csv(path)                                              # read csv
    df['date'] = pd.to_datetime(df['date'])                             # parse dates
    df = df.sort_values(['machine_id', 'date']).reset_index(drop=True)  # sort per machine
    df['is_failure'] = df['failure_type'].isin(
        CONFIG['failure_types']
    ).astype(int)                                                       # 1 = failure log
    return df


# ─────────────────────────────────────
# 3. BUILD SEQUENCE SUMMARIES
# ─────────────────────────────────────

def build_rows(df):
    rows = []
    seq_len = CONFIG['sequence_len']
    horizon = CONFIG['horizon']

    for machine_id, g in df.groupby('machine_id'):
        g = g.reset_index(drop=True)                                    # per-machine logs

        for i in range(len(g) - seq_len - horizon + 1):
            window = g.iloc[i:i + seq_len]                              # 10-log window
            future = g.iloc[i + seq_len:i + seq_len + horizon]          # next N logs
            rows.append({
                'prev_type'      : g.iloc[i + seq_len - 1]['failure_type'],
                'window_warnings': int((window['failure_type'] == CONFIG['warning_type']).sum()),
                'label'          : int(future['is_failure'].max()),     # failure within horizon
            })

    return pd.DataFrame(rows)


# ─────────────────────────────────────
# 4. REPORT FUNCTIONS
# ─────────────────────────────────────

def report_split_info(train_df, val_df):
    print("\n" + "=" * 50)
    print("SPLIT INFO")
    print("=" * 50)
    print(f"Train machines : {train_df['machine_id'].nunique()} | "
          f"{train_df['date'].min().date()} to {train_df['date'].max().date()}")
    print(f"Val machines   : {val_df['machine_id'].nunique()} | "
          f"{val_df['date'].min().date()} to {val_df['date'].max().date()}")
    shared = set(train_df['machine_id']) & set(val_df['machine_id'])
    print(f"Shared machines: {len(shared)}")


def report_label_rate(train_rows, val_rows):
    print("\n" + "=" * 50)
    print("LABEL RATE (share of sequences labeled at-risk)")
    print("=" * 50)
    print(f"Train : {train_rows['label'].mean() * 100:.2f}%")
    print(f"Val   : {val_rows['label'].mean() * 100:.2f}%")


def report_transitions(train_rows):
    print("\n" + "=" * 50)
    print("P(next log is failure | type of last log in window)")
    print("=" * 50)
    table = train_rows.groupby('prev_type')['label'].agg(['count', 'mean'])
    table['mean'] = (table['mean'] * 100).round(2)
    print(table.rename(columns={'mean': 'failure_%'}))

    print("\nP(next log is failure | warnings in window)")
    table2 = train_rows.groupby('window_warnings')['label'].agg(['count', 'mean'])
    table2['mean'] = (table2['mean'] * 100).round(2)
    print(table2.rename(columns={'mean': 'failure_%'}))


def report_baseline(train_rows, val_rows):
    print("\n" + "=" * 50)
    print("SIMPLE BASELINE (no neural network)")
    print("=" * 50)

    # failure rate per last-log type, learned on train
    rate_map  = train_rows.groupby('prev_type')['label'].mean()
    base_rate = train_rows['label'].mean()

    # predict at-risk when that type's rate beats the overall rate
    val_probs = val_rows['prev_type'].map(rate_map).fillna(base_rate)
    val_preds = (val_probs > base_rate).astype(int)
    labels    = val_rows['label']

    accuracy  = (val_preds == labels).mean() * 100
    majority  = max(labels.mean(), 1 - labels.mean()) * 100
    tp        = ((val_preds == 1) & (labels == 1)).sum()
    recall    = tp / max((labels == 1).sum(), 1) * 100
    precision = tp / max((val_preds == 1).sum(), 1) * 100

    print(f"Majority-class accuracy : {majority:.2f}%")
    print(f"Baseline accuracy       : {accuracy:.2f}%")
    print(f"Baseline at-risk recall : {recall:.2f}%")
    print(f"Baseline at-risk prec.  : {precision:.2f}%")

    if accuracy > majority + 5:
        print("\n✅ Last-log type carries real signal - the model/features can be improved")
    else:
        print("\n⚠️  Last-log type barely beats guessing - label may be unpredictable")


# ─────────────────────────────────────
# 5. MAIN
# ─────────────────────────────────────

def main():
    print("Loading splits...")
    train_df = load_split(CONFIG['train_path'])
    val_df   = load_split(CONFIG['val_path'])

    train_rows = build_rows(train_df)
    val_rows   = build_rows(val_df)

    report_split_info(train_df, val_df)
    report_label_rate(train_rows, val_rows)
    report_transitions(train_rows)
    report_baseline(train_rows, val_rows)


if __name__ == "__main__":
    main()