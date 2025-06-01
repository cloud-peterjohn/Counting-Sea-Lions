import pandas as pd
import numpy as np

df = pd.read_csv("test_result/submission.csv")

df["pups"] = df["pups"] * 1.3
df["juveniles"] = df["juveniles"] * 0.85
df["adult_females"] = df["adult_females"] * 0.96
df["adult_males"] = df["adult_males"] * 1.55
df["subadult_males"] = df["subadult_males"] * 1.2

numeric_columns = [
    "adult_males",
    "subadult_males",
    "adult_females",
    "juveniles",
    "pups",
]
for col in numeric_columns:
    df[col] = np.round(df[col]).astype(int)

df.to_csv("test_result/final_submission.csv", index=False)

print(df.head())
