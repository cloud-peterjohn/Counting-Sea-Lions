import pandas as pd

df = pd.read_csv("test_result/submissions.csv")

test_ids = set(df.iloc[:, 0])

missing_ids = [i for i in range(0, 18636) if i not in test_ids]

print("Missing test_id: ", missing_ids)
