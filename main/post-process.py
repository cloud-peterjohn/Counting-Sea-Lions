import pandas as pd
import numpy as np

df = pd.read_csv("test_result/submission.csv")

df["pups"] = df["pups"] * 1.2
df["juveniles"] = df["juveniles"] * 1.5
df["adult_females"] = df["adult_females"] * 0.5
numeric_columns = [
    "adult_males",
    "subadult_males",
    "adult_females",
    "juveniles",
    "pups",
]
for col in numeric_columns:
    df[col] = df[col] * 1.0
    df[col] = np.round(df[col]).astype(int)

df.to_csv("test_result/submission(7).csv", index=False)

print(df.head())
