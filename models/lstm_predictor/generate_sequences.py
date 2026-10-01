import os
import numpy as np
import pandas as pd

# ─────────────────────────────────────
# 1. CONFIGURATION
# ─────────────────────────────────────

CONFIG = {
    'source_path'        : 'data/clean/maintenance_logs_clean.csv',
    'output_dir'         : 'data/sequences/',
    'seed'               : 42,
    'start_date'         : '2024-01-01',
    'logs_per_machine'   : 150,
    'machines'           : {'train': 36, 'val': 12, 'test': 12},
    'healthy_len'        : (12, 30),
    'degrade_len'        : (4, 8),
    'healthy_warn_prob'  : 0.05,
    'degrade_warn_start' : 0.40,
    'degrade_warn_end'   : 0.95,
    'normal_type'        : 'No Failure',
    'warning_type'       : 'Pre-Failure Warning',
    'failure_types'      : [
        'Bearing Failure',
        'Electrical Fault',
        'Lubrication Failure',
        'Mechanical Fault',
    ],
}


# ─────────────────────────────────────
# 2. LOAD LOG POOLS
# ─────────────────────────────────────

def load_pools():
    df = pd.read_csv(CONFIG['source_path'])                             # read source logs
    df = df.dropna(subset=['clean_log', 'failure_type'])                # drop incomplete rows
    all_types = [CONFIG['normal_type'], CONFIG['warning_type']] + CONFIG['failure_types']
    pools = {}
    for t in all_types:
        pools[t] = df.loc[df['failure_type'] == t, 'clean_log'].unique().tolist()  # unique texts per type
        print(f"  ✅ {t}: {len(pools[t])} unique log texts")
        if len(pools[t]) == 0:
            print(f"  ⚠️  No logs found for {t}")
    return pools


# ─────────────────────────────────────
# 3. SIMULATE ONE MACHINE
# ─────────────────────────────────────

def sample_log(pools, log_type, rng):
    return log_type, str(rng.choice(pools[log_type]))                   # random text of that type


def simulate_machine(machine_id, pools, rng):
    types = []

    while len(types) < CONFIG['logs_per_machine']:
        # healthy phase: mostly normal logs, rare warnings
        n_healthy = rng.integers(CONFIG['healthy_len'][0], CONFIG['healthy_len'][1] + 1)
        for _ in range(n_healthy):
            is_warn = rng.random() < CONFIG['healthy_warn_prob']
            types.append(CONFIG['warning_type'] if is_warn else CONFIG['normal_type'])

        # degradation phase: warning probability ramps up
        n_degrade = rng.integers(CONFIG['degrade_len'][0], CONFIG['degrade_len'][1] + 1)
        probs = np.linspace(CONFIG['degrade_warn_start'], CONFIG['degrade_warn_end'], n_degrade)
        for p in probs:
            types.append(CONFIG['warning_type'] if rng.random() < p else CONFIG['normal_type'])

        # failure event
        types.append(str(rng.choice(CONFIG['failure_types'])))

    types = types[:CONFIG['logs_per_machine']]                          # trim to fixed length

    dates = pd.date_range(CONFIG['start_date'], periods=len(types), freq='D')  # one log per day
    rows = []
    for date, log_type in zip(dates, types):
        ftype, text = sample_log(pools, log_type, rng)
        rows.append({
            'date'        : date,
            'machine_id'  : machine_id,
            'failure_type': ftype,
            'clean_log'   : text,
        })
    return rows


# ─────────────────────────────────────
# 4. BUILD ONE SPLIT
# ─────────────────────────────────────

def build_split(name, n_machines, start_index, pools, rng):
    rows = []
    for i in range(n_machines):
        machine_id = f"M-{start_index + i:03d}"                         # globally unique machine id
        rows.extend(simulate_machine(machine_id, pools, rng))
    df = pd.DataFrame(rows)

    path = os.path.join(CONFIG['output_dir'], f"{name}.csv")
    df.to_csv(path, index=False)                                        # save split csv

    failure_share = df['failure_type'].isin(CONFIG['failure_types']).mean() * 100
    print(f"  ✅ {name}: {n_machines} machines, {len(df)} logs, "
          f"{failure_share:.1f}% failure logs -> {path}")
    return start_index + n_machines


# ─────────────────────────────────────
# 5. MAIN
# ─────────────────────────────────────

def main():
    os.makedirs(CONFIG['output_dir'], exist_ok=True)                    # ensure output folder
    rng = np.random.default_rng(CONFIG['seed'])                         # seeded generator

    print("Loading log pools...")
    pools = load_pools()

    print("\nGenerating machine-level splits...")
    start_index = 1
    for name, n_machines in CONFIG['machines'].items():
        start_index = build_split(name, n_machines, start_index, pools, rng)

    print("\n✅ Done. Splits contain different machines, so val/test machines are unseen.")


if __name__ == "__main__":
    main()