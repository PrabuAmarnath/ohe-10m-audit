import pandas as pd
from ohe_engine import OHEInspectionEngine

csv_path = r"D:\ohe_inspection\ohe_outputs\synthetic_ohe_600k.csv"
df = pd.read_csv(csv_path)

eng = OHEInspectionEngine()
df_proc, _ = eng.process_dataframe(df)
rep_10m = eng.generate_mast_report(df_proc, step_m=10.0, save_excel=True)

print("Total 10m rows:", len(rep_10m))
print("\nWires_Seen counts:\n", rep_10m["Wires_Seen"].value_counts().sort_index())
print("\nTop Overlap_Status counts:\n", rep_10m["Overlap_Status"].value_counts().head(12))

io_mask = rep_10m["Overlap_Status"].str.contains("IO (>=50cm)", regex=False)
io_rows = rep_10m[io_mask]
print("\nSample IO (>=50cm) rows:", len(io_rows))
if len(io_rows) > 0:
    cols = [
        "Chainage_Window_m",
        "Track_Alignment",
        "Outgoing_OHE_Stagger_cm",
        "Incoming_OHE_Stagger_cm",
        "Outgoing_Incoming_Separation_cm",
        "Overlap_Status",
    ]
    print(io_rows[cols].head(10).to_string())