from pendulum import DoublePendulum
import numpy as np
import pyarrow.parquet as pq
import pyarrow as pa
import os

PARQUET_FILE = "test_pendulum2.parquet"
T   = 10.0    # seconds — long enough to expose any drift
DT  = 0.01
ENERGY_TOL = 1e-3   # 0.1% max allowed relative drift for PASS

# Wide range of starting conditions covering all physical regimes (degrees)
CONDITIONS = [
    ( 10,   0,  "small"),           # near-linear, easy case
    ( 30,  30,  "moderate"),        # moderate amplitude
    ( 45,  90,  "asymmetric"),      # asymmetric medium
    ( 90,   0,  "quarter_t1"),      # large t1, t2 at rest
    ( 90,  90,  "both_90"),         # both at 90 deg
    (120, 120,  "large"),           # large amplitude both
    (150,  45,  "near_inv_t1"),     # t1 near inverted
    (150, 270,  "near_inv_asym"),   # asymmetric near-inverted
    (170,  90,  "chaotic_1"),       # highly chaotic
    (170, 270,  "chaotic_2"),       # hardest case — both near inverted
]


def compute_energy(df):
    """Total mechanical energy from a sim DataFrame."""
    m1 = float(df["m1"].iloc[0]); m2 = float(df["m2"].iloc[0])
    l1 = float(df["l1"].iloc[0]); l2 = float(df["l2"].iloc[0])
    g  = float(df["g"].iloc[0])
    t1 = df["theta1"].to_numpy();     t2 = df["theta2"].to_numpy()
    o1 = df["theta1_dot"].to_numpy(); o2 = df["theta2_dot"].to_numpy()
    delta = t1 - t2
    KE = (0.5*(m1+m2)*l1**2*o1**2
          + m2*l1*l2*o1*o2*np.cos(delta)
          + 0.5*m2*l2**2*o2**2)
    PE = -(m1+m2)*g*l1*np.cos(t1) - m2*g*l2*np.cos(t2)
    return KE + PE


def main():
    # Remove stale parquet so ParquetWriter starts fresh
    if os.path.exists(PARQUET_FILE):
        os.remove(PARQUET_FILE)
        print(f"Removed existing {PARQUET_FILE}\n")

    writer  = None
    summary = []

    print(f"Running {len(CONDITIONS)} simulations  (T={T}s, dt={DT}s)\n")
    print(f"{'SIM':>8}  {'t1_deg':>6} {'t2_deg':>6}  {'LABEL':<18}  {'max|dE/E0|':>12}  STATUS")
    print("-" * 72)

    for i, (t1_deg, t2_deg, label) in enumerate(CONDITIONS):
        sim_id = f"sim{i:06d}"
        t1_rad = np.radians(t1_deg)
        t2_rad = np.radians(t2_deg)

        pendulum = DoublePendulum(t1_rad, t2_rad, T=T, dt=DT)
        df = pendulum.generateTimeData()

        # Attach metadata so the visualiser can reconstruct parameters
        df["sim_id"]          = sim_id
        df["run_id"]          = i
        df["label"]           = label
        df["init_theta1_deg"] = float(t1_deg)
        df["init_theta2_deg"] = float(t2_deg)

        # Energy conservation check
        # Normalise by max possible PE (both bobs at top) so E0≈0 configs don't blow up
        m1_v = float(df["m1"].iloc[0]); m2_v = float(df["m2"].iloc[0])
        l1_v = float(df["l1"].iloc[0]); l2_v = float(df["l2"].iloc[0])
        g_v  = float(df["g"].iloc[0])
        E_scale   = max(abs(float(compute_energy(df)[0])),
                        (m1_v + m2_v) * g_v * l1_v + m2_v * g_v * l2_v)
        E         = compute_energy(df)
        E0        = E[0]
        rel_drift = (E - E0) / E_scale
        max_drift = float(np.max(np.abs(rel_drift)))
        status    = "PASS" if max_drift < ENERGY_TOL else "FAIL"
        summary.append((sim_id, t1_deg, t2_deg, label, max_drift, status))

        print(f"{sim_id:>8}  {t1_deg:>5} {t2_deg:>5}  {label:<18}  {max_drift:>12.2e}  {status}")

        # Write to Parquet
        table = pa.Table.from_pandas(df)
        if writer is None:
            writer = pq.ParquetWriter(PARQUET_FILE, table.schema)
        writer.write_table(table)

    if writer:
        writer.close()

    # Final summary
    n_pass = sum(1 for r in summary if r[5] == "PASS")
    n_fail = len(summary) - n_pass
    print("-" * 72)
    print(f"\n{len(summary)} simulations  |  {n_pass} PASS  |  {n_fail} FAIL")
    if n_fail == 0:
        print("PASS: Energy conserved across all starting conditions.")
    else:
        print("FAIL: Energy drift exceeded tolerance in some cases.")
    print(f"\nParquet written -> {PARQUET_FILE}")
    print(f"\nNext: python pendulumVisualization.py")


if __name__ == "__main__":
    main()