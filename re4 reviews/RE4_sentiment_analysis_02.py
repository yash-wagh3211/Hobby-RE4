"""
re4_sentiment_nlptown.py
──────────────────────────────────────────────────────────────────────────────
Sentiment analysis using nlptown/bert-base-multilingual-uncased-sentiment.

Model output  : star rating label  →  "1 star" … "5 stars"
Columns added : Sentiment_Label    →  "1 star" … "5 stars"
                Star_Rating        →  integer 1–5
                Confidence_Score   →  float 0.0–1.0
                Sentiment_Category →  Negative / Neutral / Positive
                Model_Agrees       →  bool (vs Steam voted_up)

Input  : re4_reviews_clean.csv
Output : re4_reviews_sentiment.csv
Plots  : re4_sentiment_plots.png

Install:
  pip install pandas numpy matplotlib seaborn transformers torch

Run:
  python re4_sentiment_nlptown.py
"""

# ── Imports ───────────────────────────────────────────────────────────────────
import re
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from transformers import pipeline

warnings.filterwarnings("ignore")
sns.set_theme(style="darkgrid", palette="deep")

# ═════════════════════════════════════════════════════════════════════════════
# 0.  CONFIGURATION
# ═════════════════════════════════════════════════════════════════════════════
INPUT_CSV   = "re4_reviews_clean.csv"
OUTPUT_CSV  = "re4_reviews_sentiment_02.csv"
TEXT_COL    = "Complete english only reviews"

MODEL_NAME  = "nlptown/bert-base-multilingual-uncased-sentiment"
BATCH_SIZE  = 16      # safe for CPU; raise to 32–64 on GPU
MAX_LENGTH  = 512     # BERT hard limit
SAMPLE_ROWS = 10

# Star → sentiment bucket mapping
STAR_TO_CATEGORY = {
    1: "Negative",
    2: "Negative",
    3: "Neutral",
    4: "Positive",
    5: "Positive",
}

# Colour palette (consistent across every plot)
PALETTE = {
    "Negative" : "#C44E52",
    "Neutral"  : "#CCB974",
    "Positive" : "#4C72B0",
}
STAR_PALETTE = {
    "1 star"  : "#C44E52",
    "2 stars" : "#E07A5F",
    "3 stars" : "#CCB974",
    "4 stars" : "#6BAF8D",
    "5 stars" : "#4C72B0",
}

# ═════════════════════════════════════════════════════════════════════════════
# STEP 1 — LOAD & VALIDATE CSV
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "═"*65)
print("  RE4 STEAM REVIEWS — NLPTown Sentiment Pipeline")
print("═"*65)
print("\n[1/5]  Loading CSV ...")

df = pd.read_csv(INPUT_CSV)

# Fix boolean columns saved as "TRUE"/"FALSE" strings by Excel/CSV
for col in ["voted_up", "is_zero_playtime", "is_outlier_len"]:
    if col in df.columns:
        df[col] = (
            df[col].astype(str).str.strip().str.upper()
            .map({"TRUE": True, "FALSE": False, "1": True, "0": False})
        )

if TEXT_COL not in df.columns:
    raise ValueError(
        f"Column '{TEXT_COL}' not found.\nAvailable: {list(df.columns)}"
    )

df[TEXT_COL] = df[TEXT_COL].astype(str).fillna("").str.strip()
before       = len(df)
df           = df[df[TEXT_COL].ne("")].reset_index(drop=True)

print(f"   Rows loaded     : {before:,}")
print(f"   Empty dropped   : {before - len(df)}")
print(f"   Ready to score  : {len(df):,} reviews")

# ═════════════════════════════════════════════════════════════════════════════
# STEP 2 — LOAD MODEL
# ═════════════════════════════════════════════════════════════════════════════
print(f"\n[2/5]  Loading model: {MODEL_NAME}")
print("       First run downloads ~700 MB; cached on subsequent runs ...")

analyzer = pipeline(
    task       = "sentiment-analysis",
    model      = MODEL_NAME,
    truncation = True,
    max_length = MAX_LENGTH,
    device     = -1,       # -1 = CPU  |  0 = first CUDA GPU
)
print("       Model loaded successfully.")

# ═════════════════════════════════════════════════════════════════════════════
# STEP 3 — BATCH PROCESSING WITH SAFETY NET
# ═════════════════════════════════════════════════════════════════════════════
print(f"\n[3/5]  Scoring {len(df):,} reviews (batch size = {BATCH_SIZE}) ...")

texts         = df[TEXT_COL].tolist()
labels_out    = []
scores_out    = []
error_log     = []
total_batches = (len(texts) + BATCH_SIZE - 1) // BATCH_SIZE

for batch_num, start in enumerate(range(0, len(texts), BATCH_SIZE), 1):
    batch = texts[start : start + BATCH_SIZE]

    try:
        # ── happy path ────────────────────────────────────────────────────
        results = analyzer(batch)
        for r in results:
            labels_out.append(r["label"])           # e.g. "4 stars"
            scores_out.append(round(r["score"], 4))

    except Exception as batch_err:
        # ── batch failed → retry row-by-row ──────────────────────────────
        print(f"   [!] Batch {batch_num} failed — retrying row by row ...")
        for row_idx, text in enumerate(batch, start=start):
            try:
                r = analyzer([text])[0]
                labels_out.append(r["label"])
                scores_out.append(round(r["score"], 4))
            except Exception as row_err:
                error_log.append({
                    "row_index"    : row_idx,
                    "text_preview" : text[:80],
                    "error"        : str(row_err),
                })
                labels_out.append("ERROR")
                scores_out.append(None)

    # progress every 10 batches
    if batch_num % 10 == 0 or batch_num == total_batches:
        done = min(start + BATCH_SIZE, len(texts))
        pct  = batch_num / total_batches * 100
        print(f"   Batch {batch_num:>4}/{total_batches}  |  "
              f"{done:>5,}/{len(texts):,}  |  {pct:.1f}%")

if error_log:
    pd.DataFrame(error_log).to_csv("re4_sentiment_errors.csv", index=False)
    print(f"\n   {len(error_log)} error(s) logged → re4_sentiment_errors.csv")
else:
    print("\n   No errors encountered.")

# ═════════════════════════════════════════════════════════════════════════════
# STEP 4 — BUILD RESULT COLUMNS
# ═════════════════════════════════════════════════════════════════════════════
print("\n[4/5]  Structuring output columns ...")

df["Sentiment_Label"]    = labels_out                          # "1 star" … "5 stars"
df["Confidence_Score"]   = pd.to_numeric(scores_out, errors="coerce")

# Extract integer star rating from label string  ("4 stars" → 4)
def parse_star(label: str) -> int | None:
    m = re.search(r"\d", str(label))
    return int(m.group()) if m else None

df["Star_Rating"]        = df["Sentiment_Label"].map(parse_star).astype("Int64")

# Bucket into Negative / Neutral / Positive
df["Sentiment_Category"] = df["Star_Rating"].map(STAR_TO_CATEGORY)

# Agreement: does the model's category match Steam's voted_up?
if "voted_up" in df.columns:
    steam_map             = {True: "Positive", False: "Negative"}
    df["Steam_Sentiment"] = df["voted_up"].map(steam_map)
    df["Model_Agrees"]    = df["Sentiment_Category"] == df["Steam_Sentiment"]
    agree_pct             = df["Model_Agrees"].mean() * 100
    print(f"   Model vs Steam agreement : {agree_pct:.1f}%")
    df.drop(columns=["Steam_Sentiment"], inplace=True)
else:
    df["Model_Agrees"] = np.nan
    agree_pct          = 0.0

# ── Verification sample ───────────────────────────────────────────────────────
print(f"\n   — {SAMPLE_ROWS}-row random verification sample —")
show_cols = [TEXT_COL, "voted_up", "Sentiment_Label", "Star_Rating",
             "Confidence_Score", "Sentiment_Category", "Model_Agrees"]
show_cols = [c for c in show_cols if c in df.columns]
pd.set_option("display.max_colwidth", 60)
pd.set_option("display.width", 220)
print(df[show_cols].sample(SAMPLE_ROWS, random_state=42).to_string(index=True))

# ═════════════════════════════════════════════════════════════════════════════
# STEP 5 — SEABORN PLOTS (6-panel grid)
# ═════════════════════════════════════════════════════════════════════════════
print("\n[5/5]  Generating plots ...")

fig = plt.figure(figsize=(18, 14))
fig.suptitle(
    "RE4 Steam Reviews — NLPTown Sentiment Analysis\n"
    f"(model: {MODEL_NAME}  |  n = {len(df):,} reviews)",
    fontsize=14, fontweight="bold", y=0.98
)
gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])
ax3 = fig.add_subplot(gs[0, 2])
ax4 = fig.add_subplot(gs[1, 0])
ax5 = fig.add_subplot(gs[1, 1])
ax6 = fig.add_subplot(gs[1, 2])
ax7 = fig.add_subplot(gs[2, :])   # wide bottom row

# ── Plot 1: Star rating distribution (countplot) ─────────────────────────────
star_order = ["1 star", "2 stars", "3 stars", "4 stars", "5 stars"]
star_order_present = [s for s in star_order if s in df["Sentiment_Label"].values]
sns.countplot(
    data=df, x="Sentiment_Label", order=star_order_present,
    palette=STAR_PALETTE, ax=ax1
)
ax1.set_title("Star Rating Distribution", fontweight="bold")
ax1.set_xlabel("Model Rating")
ax1.set_ylabel("Count")
ax1.tick_params(axis="x", rotation=20)
for p in ax1.patches:
    ax1.annotate(f"{int(p.get_height()):,}",
                 (p.get_x() + p.get_width() / 2, p.get_height() + 2),
                 ha="center", va="bottom", fontsize=8)

# ── Plot 2: Sentiment category (Neg / Neutral / Pos) ─────────────────────────
cat_order = ["Negative", "Neutral", "Positive"]
sns.countplot(
    data=df, x="Sentiment_Category", order=cat_order,
    palette=PALETTE, ax=ax2
)
ax2.set_title("Sentiment Category", fontweight="bold")
ax2.set_xlabel("Category")
ax2.set_ylabel("Count")
for p in ax2.patches:
    ax2.annotate(f"{int(p.get_height()):,}",
                 (p.get_x() + p.get_width() / 2, p.get_height() + 2),
                 ha="center", va="bottom", fontsize=8)

# ── Plot 3: Model vs Steam vote agreement (pie) ───────────────────────────────
if "Model_Agrees" in df.columns and df["Model_Agrees"].notna().any():
    agree_counts = df["Model_Agrees"].value_counts()
    pie_labels   = ["Agree" if k else "Disagree" for k in agree_counts.index]
    pie_colors   = ["#55A868" if k else "#C44E52" for k in agree_counts.index]
    ax3.pie(
        agree_counts.values,
        labels=pie_labels,
        colors=pie_colors,
        autopct="%1.1f%%",
        startangle=90,
        textprops={"fontsize": 9},
    )
    ax3.set_title(f"Model vs Steam Vote\n({agree_pct:.1f}% agreement)", fontweight="bold")
else:
    ax3.set_visible(False)

# ── Plot 4: Confidence score by star rating (boxplot) ────────────────────────
sns.boxplot(
    data=df[df["Sentiment_Label"] != "ERROR"],
    x="Sentiment_Label", y="Confidence_Score",
    order=star_order_present,
    palette=STAR_PALETTE, ax=ax4
)
ax4.set_title("Confidence by Star Rating", fontweight="bold")
ax4.set_xlabel("Star Rating")
ax4.set_ylabel("Confidence Score")
ax4.tick_params(axis="x", rotation=20)

# ── Plot 5: Confidence score by sentiment category (violin) ──────────────────
valid_cats = [c for c in cat_order if c in df["Sentiment_Category"].values]
sns.violinplot(
    data=df[df["Sentiment_Category"].isin(valid_cats)],
    x="Sentiment_Category", y="Confidence_Score",
    order=valid_cats, palette=PALETTE,
    inner="quartile", ax=ax5
)
ax5.set_title("Confidence Distribution\nby Category", fontweight="bold")
ax5.set_xlabel("Category")
ax5.set_ylabel("Confidence Score")

# ── Plot 6: Star rating by Steam vote (stacked count) ────────────────────────
if "voted_up" in df.columns:
    cross = (
        df.groupby(["Sentiment_Label", "voted_up"])
        .size().reset_index(name="count")
    )
    cross["voted_up_str"] = cross["voted_up"].map(
        {True: "Steam Positive", False: "Steam Negative"}
    )
    pivot = cross.pivot(
        index="Sentiment_Label", columns="voted_up_str", values="count"
    ).reindex(star_order_present).fillna(0)
    pivot.plot(
        kind="bar", stacked=True,
        color=["#C44E52", "#4C72B0"],
        ax=ax6, legend=True
    )
    ax6.set_title("Star Rating vs Steam Vote", fontweight="bold")
    ax6.set_xlabel("Star Rating")
    ax6.set_ylabel("Count")
    ax6.tick_params(axis="x", rotation=20)
    ax6.legend(fontsize=8, title="Steam Vote", title_fontsize=8)
else:
    ax6.set_visible(False)

# ── Plot 7: Confidence score KDE by category (wide bottom panel) ─────────────
for cat in valid_cats:
    sub = df[df["Sentiment_Category"] == cat]["Confidence_Score"].dropna()
    sns.kdeplot(sub, label=cat, color=PALETTE[cat], fill=True, alpha=0.25, ax=ax7)
ax7.set_title("Confidence Score Density by Sentiment Category", fontweight="bold")
ax7.set_xlabel("Confidence Score")
ax7.set_ylabel("Density")
ax7.legend(title="Category", fontsize=9)

plt.savefig("re4_sentiment_plots.png", dpi=150, bbox_inches="tight")
plt.show()
print("   Plots saved → re4_sentiment_plots.png")

# ═════════════════════════════════════════════════════════════════════════════
# SAVE OUTPUT CSV
# ═════════════════════════════════════════════════════════════════════════════
df.to_csv(OUTPUT_CSV, index=False)

print(f"\n{'═'*65}")
print(f"  OUTPUT SAVED → {OUTPUT_CSV}")
print(f"  Shape : {df.shape[0]:,} rows × {df.shape[1]} columns")
print(f"\n  New columns added:")
print(f"    Sentiment_Label    — raw model output  (e.g. '4 stars')")
print(f"    Star_Rating        — integer 1–5")
print(f"    Confidence_Score   — model confidence  (0.0 – 1.0)")
print(f"    Sentiment_Category — Negative / Neutral / Positive")
print(f"    Model_Agrees       — matches Steam voted_up?")
print(f"\n  Star rating breakdown:")
print(df["Sentiment_Label"].value_counts().reindex(star_order).dropna().to_string())
print(f"\n  Category breakdown:")
print(df["Sentiment_Category"].value_counts().reindex(cat_order).dropna().to_string())
print(f"{'═'*65}\n")