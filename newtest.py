#!/usr/bin/env python3
"""
Debug script to find the correct sum for Quiz #3
"""

import requests
import pandas as pd
import io

EMAIL = "21f3002378@ds.study.iitm.ac.in"
CSV_URL = "https://tds-llm-analysis.s-anand.net/demo-audio-data.csv"
CUTOFF = 58886

print("Downloading CSV (with timeout)...")
try:
    response = requests.get(CSV_URL, timeout=30)
    response.raise_for_status()
except requests.exceptions.Timeout:
    print("❌ Request timed out")
    exit(1)
except Exception as e:
    print(f"❌ Error: {e}")
    exit(1)

print(f"✓ Downloaded {len(response.content)} bytes")

try:
    df = pd.read_csv(io.StringIO(response.text))
    print(f"✓ CSV parsed successfully")
except Exception as e:
    print(f"❌ Failed to parse CSV: {e}")
    exit(1)

print(f"\nDataFrame Info:")
print(f"Shape: {df.shape}")
print(f"Columns: {list(df.columns)}")
print(f"\nFirst 5 rows:")
print(df.head())

col = df.columns[0]
print(f"\n" + "="*60)
print(f"Column: {col}")
print(f"Data type: {df[col].dtype}")
print(f"Min: {df[col].min()}")
print(f"Max: {df[col].max()}")
print(f"Mean: {df[col].mean():.2f}")
print(f"Total sum: {df[col].sum()}")

print(f"\n" + "="*60)
print(f"CUTOFF ANALYSIS (cutoff = {CUTOFF}):")
print(f"="*60)

# Convert to numeric if needed
df[col] = pd.to_numeric(df[col], errors='coerce')

# Try different operations
print(f"\n{col} > {CUTOFF}:")
gt_count = len(df[df[col] > CUTOFF])
gt_sum = df[df[col] > CUTOFF][col].sum()
print(f"  Rows: {gt_count}")
print(f"  Sum: {int(gt_sum) if not pd.isna(gt_sum) else 'N/A'}")

print(f"\n{col} >= {CUTOFF}:")
gte_count = len(df[df[col] >= CUTOFF])
gte_sum = df[df[col] >= CUTOFF][col].sum()
print(f"  Rows: {gte_count}")
print(f"  Sum: {int(gte_sum) if not pd.isna(gte_sum) else 'N/A'}")

print(f"\n{col} < {CUTOFF}:")
lt_count = len(df[df[col] < CUTOFF])
lt_sum = df[df[col] < CUTOFF][col].sum()
print(f"  Rows: {lt_count}")
print(f"  Sum: {int(lt_sum) if not pd.isna(lt_sum) else 'N/A'}")

print(f"\n" + "="*60)
print(f"ANSWER CANDIDATES:")
print(f"="*60)
print(f"1. Sum of rows > {CUTOFF}: {int(gt_sum) if not pd.isna(gt_sum) else 'N/A'}")
print(f"2. Sum of rows >= {CUTOFF}: {int(gte_sum) if not pd.isna(gte_sum) else 'N/A'}")
print(f"3. Sum of ALL values: {int(df[col].sum())}")
print(f"4. Count of rows > {CUTOFF}: {gt_count}")
print(f"5. Count of rows >= {CUTOFF}: {gte_count}")
print(f"6. Total rows: {len(df)}")