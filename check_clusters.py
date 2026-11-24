import pandas as pd

csv_path = r"D:\ohe_inspection\ohe_outputs\synthetic_ohe_600k.csv"
df = pd.read_csv(csv_path)

print("Unique Track_ID:", sorted(df["Track_ID"].dropna().unique().tolist()))
print("Unique Cluster_ID:", sorted(df["Cluster_ID"].dropna().unique().tolist()))
print("Wire_Gap_mm min/median/max:", df["Wire_Gap_mm"].min(), df["Wire_Gap_mm"].median(), df["Wire_Gap_mm"].max())

# Check how many rows share the same DateTime or CumDist_m (multiple detections/wires at one spot)
dt_counts = df.groupby("DateTime")["Cluster_ID"].nunique()
print("Max unique Cluster_IDs per DateTime:", dt_counts.max())
print("DateTime cluster count distribution:\n", dt_counts.value_counts().head(10))