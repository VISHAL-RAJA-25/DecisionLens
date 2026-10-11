import pandas as pd
from pathlib import Path

RAW = Path("data/raw")

for f in sorted(RAW.glob("*.csv")):
    df = pd.read_csv(f)
    print("=" * 70)
    print(f"{f.name}  ->  {df.shape[0]:,} rows, {df.shape[1]} columns")

    print("\nColumn types:")
    print(df.dtypes)

    nulls = df.isnull().sum()
    nulls = nulls[nulls > 0]
    print("\nMissing values:")
    print(nulls if len(nulls) else "None")

    print("\nDuplicate rows:", df.duplicated().sum())
    print("\nFirst 3 rows:")
    print(df.head(3).to_string())
    print()
    