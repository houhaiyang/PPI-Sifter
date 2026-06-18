# 在终端快速执行，生成 case_study.csv（100 正 + 100 负，按 prob 排序）
import pandas as pd
df = pd.read_csv("data/BIOGRID/pairs/protein_disjoint/test.csv")
pos = df[df["label"]==1].sample(200, random_state=42)
neg = df[df["label"]==0].sample(200, random_state=42)
pd.concat([pos, neg]).to_csv("data/BIOGRID/pairs/protein_disjoint/case_study.csv", index=False)

