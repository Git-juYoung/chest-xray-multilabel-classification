import pandas as pd

df = pd.read_csv('label_chexpert.csv')

print(df.shape, '\n')
print(df.columns.tolist(), '\n')

print("n_subjects:", df["subject_id"].nunique(), '\n')

print("images per subject:\n", df["subject_id"].value_counts(), '\n')

label_cols = df.columns[5:]
print("label columns:\n", label_cols.tolist(), '\n')

print("label value distribution:\n",
      df[label_cols].apply(lambda x: x.value_counts(dropna=False)), '\n')

pos_rate = (df[label_cols] == 1.0).mean()
neg_rate = (df[label_cols] == 0.0).mean()
uncertain_rate = (df[label_cols] == -1.0).mean()
nan_rate = df[label_cols].isna().mean()

print("positive rate:\n", pos_rate, '\n')
print("negative rate:\n", neg_rate, '\n')
print("uncertain(-1) rate:\n", uncertain_rate, '\n')
print("nan rate:\n", nan_rate, '\n')

pos = (df[label_cols] == 1.0).sum()
neg = (df[label_cols] == 0.0).sum()

pn = pos + neg

pn_ratio = pd.DataFrame({
    "positive": pos,
    "negative": neg,
    "positive_ratio": pos / pn,
    "negative_ratio": neg / pn,
})

print(pn_ratio.sort_values("positive_ratio"))

df.head(2)

