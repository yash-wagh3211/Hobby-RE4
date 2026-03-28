"""
re4_sentiment_run.py
──────────────────────────────────────────────────────────────────────────────
Runs HuggingFace sentiment analysis directly on re4_reviews_clean.csv.

CSV columns detected:
  id, voted_up, playtime_forever, lang, is_zero_playtime,
  is_outlier_len, voted_up_str, Complete english only reviews

Output: re4_reviews_sentiment.csv + re4_sentiment_qa.png

Install once:
  pip install pandas numpy matplotlib seaborn transformers torch
Run:
  python re4_sentiment_run.py
"""

import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import pipeline

warnings.filterwarnings("ignore")

# ═════════════════════════════════════════════════════════════════════════════
# CONFIG — change paths here if needed, nothing else required
# ═════════════════════════════════════════════════════════════════════════════
INPUT_CSV    = "re4_reviews_clean.csv"
OUTPUT_CSV   = "re4_reviews_sentiment.csv"
TEXT_COL     = "Complete english only reviews"   # exact column in your CSV
MODEL_NAME   = "distilbert-base-uncased-finetuned-sst-2-english"
BATCH_SIZE   = 16     # safe for CPU; raise to 32-64 if you have a GPU
MAX_LENGTH   = 512    # DistilBERT token limit; longer reviews are truncated
SAMPLE_ROWS  = 10     # rows shown in verification printout

# ═════════════════════════════════════════════════════════════════════════════
# STEP 1 — LOAD CSV
# ═════════════════════════════════════════════════════════════════════════════
print("\n── STEP 1 / 5  Loading CSV ──────────────────────────────────────────")

df = pd.read_csv(INPUT_CSV)

# Fix boolean columns stored as "TRUE"/"FALSE" strings by Excel/CSV export
for col in ["voted_up", "is_zero_playtime", "is_outlier_len"]:
    if col in df.columns:
        df[col] = df[col].astype(str).str.strip().str.upper().map(
            {"TRUE": True, "FALSE": False, "1": True, "0": False}
        ).fillna(df[col])

if TEXT_COL not in df.columns:
    raise ValueError(
        f"Column '{TEXT_COL}' not found.\nAvailable: {list(df.columns)}"
    )

# Clean review text — never send NaN or blank strings to the model
df[TEXT_COL] = df[TEXT_COL].astype(str).fillna("").str.strip()
before       = len(df)
df           = df[df[TEXT_COL].ne("")].reset_index(drop=True)

print(f"   Rows loaded  : {before:,}")
print(f"   Empty dropped: {before - len(df)}")
print(f"   Ready to score: {len(df):,} reviews")

# ═════════════════════════════════════════════════════════════════════════════
# STEP 2 — LOAD MODEL
# ═════════════════════════════════════════════════════════════════════════════
print("\n── STEP 2 / 5  Loading HuggingFace model ───────────────────────────")
print(f"   {MODEL_NAME}")
print("   First run downloads ~250 MB and caches it locally.")
print("   Subsequent runs load from cache instantly.\n")

analyzer = pipeline(
    task       = "sentiment-analysis",
    model      = MODEL_NAME,
    truncation = True,
    max_length = MAX_LENGTH,
    device     = -1,     # -1 = CPU  |  0 = CUDA GPU
)
print("   Model ready.")

# ═════════════════════════════════════════════════════════════════════════════
# STEP 3 — BATCH PROCESSING WITH SAFETY NET
# ═════════════════════════════════════════════════════════════════════════════
print("\n── STEP 3 / 5  Scoring reviews ─────────────────────────────────────")

texts         = df[TEXT_COL].tolist()
labels        = []
scores        = []
error_log     = []
total_batches = (len(texts) + BATCH_SIZE - 1) // BATCH_SIZE

for batch_num, start in enumerate(range(0, len(texts), BATCH_SIZE), 1):
    batch = texts[start : start + BATCH_SIZE]

    try:
        # --- happy path: whole batch succeeds --------------------------------
        results = analyzer(batch)
        for r in results:
            labels.append(r["label"])
            scores.append(round(r["score"], 4))

    except Exception as batch_err:
        # --- batch failed: retry one row at a time ---------------------------
        print(f"   [!] Batch {batch_num} failed ({batch_err}). "
              f"Retrying {len(batch)} rows individually ...")
        for row_idx, text in enumerate(batch, start=start):
            try:
                r = analyzer([text])[0]
                labels.append(r["label"])
                scores.append(round(r["score"], 4))
            except Exception as row_err:
                error_log.append({
                    "row_index": row_idx,
                    "text_preview": text[:80],
                    "error": str(row_err),
                })
                labels.append("ERROR")
                scores.append(None)

    # progress every 10 batches
    if batch_num % 10 == 0 or batch_num == total_batches:
        done = min(start + BATCH_SIZE, len(texts))
        pct  = batch_num / total_batches * 100
        print(f"   Batch {batch_num:>4}/{total_batches}  |  "
              f"{done:>5,}/{len(texts):,} reviews  |  {pct:.1f}%")

if error_log:
    err_df = pd.DataFrame(error_log)
    err_df.to_csv("re4_sentiment_errors.csv", index=False)
    print(f"\n   {len(error_log)} row(s) failed — saved to re4_sentiment_errors.csv")
else:
    print("\n   No errors.")

# ═════════════════════════════════════════════════════════════════════════════
# STEP 4 — ATTACH RESULTS + VERIFY
# ═════════════════════════════════════════════════════════════════════════════
print("\n── STEP 4 / 5  Attaching results ───────────────────────────────────")

df["Sentiment_Label"]  = pd.Series(labels).str.title()   # POSITIVE → Positive
df["Confidence_Score"] = pd.to_numeric(scores, errors="coerce")

# Agreement column: does the model agree with Steam's voted_up?
if "voted_up" in df.columns:
    steam_map          = {True: "Positive", False: "Negative"}
    df["steam_sent"]   = df["voted_up"].map(steam_map)
    df["model_agrees"] = df["Sentiment_Label"] == df["steam_sent"]
    agree_pct          = df["model_agrees"].mean() * 100
    print(f"   Model vs Steam vote agreement: {agree_pct:.1f}%")
    df.drop(columns=["steam_sent"], inplace=True)
else:
    df["model_agrees"] = np.nan
    agree_pct          = 0.0

# Random sample to visually sanity-check the labels
print(f"\n   — {SAMPLE_ROWS}-row random verification sample —")
show_cols = [TEXT_COL, "voted_up", "Sentiment_Label", "Confidence_Score", "model_agrees"]
show_cols = [c for c in show_cols if c in df.columns]
pd.set_option("display.max_colwidth", 72)
pd.set_option("display.width", 200)
print(df[show_cols].sample(SAMPLE_ROWS, random_state=42).to_string(index=True))

# ═════════════════════════════════════════════════════════════════════════════
# STEP 5 — PLOTS + SAVE
# ═════════════════════════════════════════════════════════════════════════════
print("\n── STEP 5 / 5  Plots & save ─────────────────────────────────────────")

PAL = {"Positive": "#4C72B0", "Negative": "#C44E52", "Error": "#AAAAAA"}
fig, axes = plt.subplots(2, 2, figsize=(13, 9))
fig.suptitle("RE4 Steam Reviews — Sentiment Analysis", fontsize=14, fontweight="bold")

# 1. Label bar chart
lc     = df["Sentiment_Label"].value_calls() if hasattr(df["Sentiment_Label"], "value_calls") else df["Sentiment_Label"].value_counts()
lc     = df["Sentiment_Label"].value_counts()
colors = [PAL.get(l, "#999") for l in lc.index]
axes[0, 0].bar(lc.index, lc.values, color=colors)
axes[0, 0].set_title("Sentiment Distribution")
axes[0, 0].set_xlabel("Label")
axes[0, 0].set_ylabel("Count")
for i, v in enumerate(lc.values):
    axes[0, 0].text(i, v + 3, f"{v:,}", ha="center", fontsize=9)

# 2. Confidence histogram by label
for lbl, col in PAL.items():
    sub = df[df["Sentiment_Label"] == lbl]["Confidence_Score"].dropna()
    if not sub.empty:
        axes[0, 1].hist(sub, bins=40, alpha=0.6, label=lbl, color=col)
axes[0, 1].set_title("Confidence Score Distribution")
axes[0, 1].set_xlabel("Score")
axes[0, 1].set_ylabel("Count")
axes[0, 1].legend()

# 3. Agreement bar chart
if "model_agrees" in df.columns and df["model_agrees"].notna().any():
    ac     = df["model_agrees"].value_counts()
    alabels = ["Agree" if k else "Disagree" for k in ac.index]
    acolors = ["#55A868" if k else "#C44E52" for k in ac.index]
    axes[1, 0].bar(alabels, ac.values, color=acolors)
    axes[1, 0].set_title(f"Model vs Steam Vote ({agree_pct:.1f}% agreement)")
    axes[1, 0].set_xlabel("Agreement")
    axes[1, 0].set_ylabel("Count")
    for i, v in enumerate(ac.values):
        axes[1, 0].text(i, v + 3, f"{v:,}", ha="center", fontsize=9)
else:
    axes[1, 0].set_visible(False)

# 4. Confidence boxplot by Steam vote
if "voted_up" in df.columns:
    df["_vl"] = df["voted_up"].map({True: "Positive Vote", False: "Negative Vote"})
    sns.boxplot(
        x="_vl", y="Confidence_Score", data=df,
        palette={"Positive Vote": "#4C72B0", "Negative Vote": "#C44E52"},
        order=["Positive Vote", "Negative Vote"], ax=axes[1, 1]
    )
    axes[1, 1].set_title("Confidence by Steam Vote")
    axes[1, 1].set_xlabel("Steam Vote")
    axes[1, 1].set_ylabel("Confidence Score")
    df.drop(columns=["_vl"], inplace=True)
else:
    axes[1, 1].set_visible(False)

plt.tight_layout()
plt.savefig("re4_sentiment_qa.png", dpi=150)
plt.show()
print("   Plots saved -> re4_sentiment_qa.png")

# Save final CSV
df.to_csv(OUTPUT_CSV, index=False)
print(f"\n   Output saved -> {OUTPUT_CSV}")
print(f"   Shape: {df.shape[0]:,} rows x {df.shape[1]} columns")
print(f"   New columns: Sentiment_Label, Confidence_Score, model_agrees")
print(f"\n   Label counts:\n{df['Sentiment_Label'].value_counts().to_string()}")    