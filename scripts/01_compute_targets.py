import os
import pandas as pd
from clcs.enrichment import build_training_table

os.makedirs("data", exist_ok=True)

meas = pd.read_csv("data/round_measurements.csv")
counts = pd.read_csv("data/round_counts.csv")

final_round = int(meas["round"].max())

train = build_training_table(meas, counts, final_round=final_round, pseudocount=1.0)
train.to_csv("data/train_table.csv", index=False)

print(f"Final round = {final_round}")
print("Wrote: data/train_table.csv")