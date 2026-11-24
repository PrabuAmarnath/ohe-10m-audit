import pandas as pd

csv_path = r"D:\ohe_inspection\ohe_outputs\synthetic_ohe_600k.csv"
df = pd.read_csv(csv_path, nrows=5000)

print("Total columns:", len(df.columns))
for i, col in enumerate(df.columns):
    print(f"{i}: {col} -> {df[col].dropna().unique()[:5]}")