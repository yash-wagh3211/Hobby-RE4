"""
Fetch the latest 2,000 English Steam reviews for Resident Evil 4 Remake (App ID: 2050650).
Saves review_text, voted_up, and playtime_forever into a Pandas DataFrame.

Steam Review API docs:
https://partner.steamgames.com/doc/store/getreviews
"""

import time
import requests
import pandas as pd

# ── Configuration ────────────────────────────────────────────────────────────
APP_ID        = 2050650       # Resident Evil 4 Remake
TARGET_COUNT  = 2000          # Total reviews to collect
BATCH_SIZE    = 100           # Max allowed per request (Steam caps at 100)
LANGUAGE      = "english"
FILTER        = "recent"      # "recent" | "updated" | "all"
REQUEST_DELAY = 1.0           # Seconds between requests (be polite to Steam)

BASE_URL = f"https://store.steampowered.com/appreviews/{APP_ID}"

# ── Fetch reviews ─────────────────────────────────────────────────────────────
def fetch_reviews(target: int = TARGET_COUNT) -> pd.DataFrame:
    records = []
    cursor  = "*"             # Steam uses cursor-based pagination
    params  = {
        "json":             1,
        "language":         LANGUAGE,
        "filter":           FILTER,
        "num_per_page":     BATCH_SIZE,
        "purchase_type":    "all",   # include both purchased & non-purchased
        "review_type":      "all",   # positive + negative
    }

    print(f"Fetching up to {target} '{LANGUAGE}' reviews for App ID {APP_ID}…\n")

    while len(records) < target:
        params["cursor"] = cursor

        response = requests.get(BASE_URL, params=params, timeout=15)
        response.raise_for_status()
        data = response.json()

        # Steam signals "no more reviews" with success=1 but an empty list
        if not data.get("success"):
            print("Steam API returned success=0. Stopping.")
            break

        batch = data.get("reviews", [])
        if not batch:
            print("No more reviews returned by Steam. Stopping early.")
            break

        for rev in batch:
            records.append({
                "review_text":       rev["review"],
                "voted_up":          rev["voted_up"],
                "playtime_forever":  rev["author"]["playtime_forever"],
            })

        fetched = len(records)
        cursor  = data.get("cursor", "")      # advance pagination cursor
        print(f"  Collected {fetched:,} / {target:,} reviews…")

        if not cursor:
            print("Cursor exhausted. No further pages available.")
            break

        time.sleep(REQUEST_DELAY)             # avoid hammering Steam

    # Trim to exactly the requested count (last batch may push us over)
    df = pd.DataFrame(records[:target])

    # ── Light type-casting ────────────────────────────────────────────────────
    df["voted_up"]         = df["voted_up"].astype(bool)
    df["playtime_forever"] = df["playtime_forever"].astype(int)   # minutes

    return df


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    df = fetch_reviews()

    print(f"\n✅ Done! DataFrame shape: {df.shape}")
    print(f"\nColumn dtypes:\n{df.dtypes}")
    print(f"\nSample rows:\n{df.head(3).to_string()}")

    # Optional: save to CSV
    out_path = "re4_reviews.csv"
    df.to_csv(out_path, index=False)
    print(f"\nSaved to '{out_path}'")
    