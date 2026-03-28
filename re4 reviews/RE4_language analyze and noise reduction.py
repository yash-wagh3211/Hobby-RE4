# Analyze re4_reviews_clean.csv for language + symbol noise patterns
import pandas as pd
import numpy as np
import re

clean_path = "re4_reviews_clean.csv"
df_clean = pd.read_csv(clean_path)

print(df_clean.head())
print(df_clean.shape)

# Basic null checks
nulls_tbl = pd.DataFrame({
    "column": df_clean.columns,
    "nulls": df_clean.isna().sum().values,
    "null_rate": (df_clean.isna().mean().values).round(4)
}).sort_values("nulls", ascending=False)
print(nulls_tbl)

# Symbol diagnostics on review_text
text_ser = df_clean["review_text"].astype("string").fillna("")

# Heuristics: non-ascii rate, digit rate, punctuation density
non_ascii_cnt = text_ser.str.count(r"[^\x00-\x7F]")
ascii_cnt = text_ser.str.len() - non_ascii_cnt
non_ascii_rate = (non_ascii_cnt / text_ser.str.len().replace(0, np.nan)).fillna(0)

punct_cnt = text_ser.str.count(r"[^\w\s]")
punct_rate = (punct_cnt / text_ser.str.len().replace(0, np.nan)).fillna(0)

# Simple English-likeness heuristic: % of a-z letters
latin_letters_cnt = text_ser.str.count(r"[A-Za-z]")
latin_rate = (latin_letters_cnt / text_ser.str.len().replace(0, np.nan)).fillna(0)

stats_tbl = pd.DataFrame({
    "metric": ["median_len", "p10_len", "p90_len", "median_non_ascii_rate", "p90_non_ascii_rate", "median_punct_rate", "p90_punct_rate", "median_latin_rate", "p10_latin_rate"],
    "value": [
        int(text_ser.str.len().median()),
        int(text_ser.str.len().quantile(0.10)),
        int(text_ser.str.len().quantile(0.90)),
        float(non_ascii_rate.median().round(4)),
        float(non_ascii_rate.quantile(0.90).round(4)),
        float(punct_rate.median().round(4)),
        float(punct_rate.quantile(0.90).round(4)),
        float(latin_rate.median().round(4)),
        float(latin_rate.quantile(0.10).round(4))
    ]
})
print(stats_tbl)

# Show examples of likely non-English / symbol-heavy rows
suspect_idx = df_clean.assign(non_ascii_rate=non_ascii_rate, punct_rate=punct_rate, latin_rate=latin_rate)

examples_non_ascii = suspect_idx.sort_values(["non_ascii_rate", "latin_rate"], ascending=[False, True]).head(8)
examples_punct = suspect_idx.sort_values("punct_rate", ascending=False).head(8)
examples_low_latin = suspect_idx.sort_values("latin_rate", ascending=True).head(8)

print(examples_non_ascii[["review_text","non_ascii_rate","latin_rate","punct_rate"]])
print(examples_punct[["review_text","non_ascii_rate","latin_rate","punct_rate"]])
print(examples_low_latin[["review_text","non_ascii_rate","latin_rate","punct_rate"]])