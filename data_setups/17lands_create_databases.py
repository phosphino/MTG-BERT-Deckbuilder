## ==========================================================
## Script: 17lands_create_databases.py
## Author: Andy Breuhaus
##
## Description:
##   - Parse 17Lands CSVs containing game outcomes
##   - Aggregate card usage across different contexts:
##       * opening hand
##       * drawn
##       * tutored
##       * hand (in hand at any point)
##   - Store results into a SQLite database
##     → One database table per input CSV file
##
## Inputs:
##   - 17Lands CSV files in:  project_root / 17Lands
##   - Scryfall JSON file:    latest oracle-cards-*.json
##
## Outputs:
##   - SQLite database file:  training_database.db
##   - Each table = one input CSV
##
## Database Schema (per table):
##   - Each row = one game
##   - Each column = one card (integer count of times seen in that game)
##   - "won" column = 1 if the player won, 0 otherwise
##
## Usage:
##   Run this script directly. No arguments needed.
##   Example:
##       python 17lands_create_databases.py
##
## Notes:
##   - Uses Polars for fast CSV parsing & aggregation
##   - Uses ADBC driver for much faster DB writes vs SQLAlchemy default
## ==========================================================


import os
import re
import gc
import polars as pl
from polars import selectors as cs
import pathlib
from collections import defaultdict
from sqlalchemy import create_engine


# ----------------------------------------------------------
# Project Paths
# ----------------------------------------------------------
project_root = pathlib.Path(r"C:/Users/breuh/OneDrive/proggy/python/MTG/roberta")

# Scryfall JSON folder (only latest file used)
scryfall_folder = project_root / "scryfall_json"
oracle_pattern = re.compile(r"oracle-cards-(\d{8})\d{6}\.json")
oracle_paths = sorted(
    [p for p in scryfall_folder.iterdir() if oracle_pattern.search(p.name)],
    key=lambda x: x.name,
    reverse=True
)
scryfall_path = oracle_paths[0]  # Latest oracle file
scryfall_pruned_path = scryfall_folder / "scryfall_pruned.json"

# 17Lands CSV folder
lands_folder = project_root / "17Lands"

# All CSVs except those containing "combo"
lands_csv_fnames = [f for f in lands_folder.glob("*.csv") if "combo" not in f.name]


# ----------------------------------------------------------
# Column Selectors
# ----------------------------------------------------------
# Substrings of card-related columns
col_filters = (
    cs.contains("opening_") |
    cs.contains("drawn_")   |
    cs.contains("tutored_") |
    cs.contains("hand_")    |
    cs.contains("won")
)

# Card-event selectors vs. outcome selector
card_selectors = cs.contains("opening_") | cs.contains("drawn_") | cs.contains("tutored_") | cs.contains("hand_")
won_selector = pl.col("won")

# Pre-scan CSVs into lazy frames, cast to integers
lazy_frames = [
    pl.scan_csv(fname).select(col_filters).with_columns(pl.all().cast(pl.Int32))
    for fname in lands_csv_fnames
]

# Prefixes used for identifying card columns
CARD_PREFIXES = ("opening_hand_", "drawn_", "tutored_")


# ----------------------------------------------------------
# Function: per_file_card_sums
# ----------------------------------------------------------
def per_file_card_sums(path, all_cards=None):
    """
    Given a 17Lands CSV file:
      - Select only card prefix columns + 'won'
      - Cast columns to compact integer types
      - Sum across prefixes → one column per card
      - Add zero columns for any missing cards (if all_cards provided)
    
    Args:
        path (Path): Path to the CSV file
        all_cards (set, optional): All card names to enforce schema consistency
    
    Returns:
        pl.LazyFrame: One column per card + 'won'
    """

    # Select only relevant columns
    lf = pl.scan_csv(path).select(cs.starts_with(CARD_PREFIXES), pl.col("won"))

    # Build mapping: card_name → list of column names
    cols = lf.collect_schema().names()
    groups = defaultdict(list)
    for c in cols:
        if c == "won":
            continue
        for p in CARD_PREFIXES:
            if c.startswith(p):
                card = c[len(p):]  # Trim prefix → card name
                groups[card].append(c)
                break

    # Cast columns to smaller int types
    to_cast = tuple({name for lst in groups.values() for name in lst})
    cast_exprs = []
    if to_cast:
        cast_exprs.append(pl.col(list(to_cast)).cast(pl.UInt16))
    if "won" in cols:
        cast_exprs.append(pl.col("won").cast(pl.UInt8))
    if cast_exprs:
        lf = lf.with_columns(*cast_exprs)

    # Horizontal sums → one column per card
    sum_exprs = [
        pl.sum_horizontal(pl.col(groups[card])).alias(card)
        for card in sorted(groups.keys())
    ]

    # Add zero columns for missing cards (optional)
    if all_cards is not None:
        missing = sorted(all_cards.difference(groups.keys()))
        if missing:
            sum_exprs += [pl.lit(0, dtype=pl.UInt16).alias(card) for card in missing]

    # Return card totals + 'won'
    out_exprs = sum_exprs + ([pl.col("won")] if "won" in cols else [])
    return lf.select(out_exprs)


# ----------------------------------------------------------
# Collect LazyFrames → Eager DataFrames
# ----------------------------------------------------------
lazyframes = (per_file_card_sums(path) for path in lands_csv_fnames)
lands_df = pl.collect_all(lazyframes)


# ----------------------------------------------------------
# Database Setup
# ----------------------------------------------------------
savename = "training_database.db"
connection_uri = f"sqlite:///{savename}"
engine = create_engine(connection_uri, future=True)

# Regex: extract (set, format) from filename
name_pattern = re.compile(r"\.(\D*)\.(\D*).csv")


# ----------------------------------------------------------
# Main Loop: Write each CSV → Database Table
# ----------------------------------------------------------
for df, name in zip(lands_df, lands_csv_fnames):
    print(f"Processing {name_pattern.search(name.name).groups()}...")
    set_, format_ = name_pattern.search(name.name).groups()
    table_name = f"{set_}_{format_}"

    print(f"Writing table {table_name}, tablesize: {df.shape}...")

    df.write_database(
        table_name=table_name,
        connection=connection_uri,
        if_table_exists="replace",
        engine="adbc"  # ADBC driver → much faster than default SQLAlchemy
    )

    print(f"Inserted {len(df)} rows into {table_name}")

    # Free memory
    del df
    gc.collect()

print("\nDatabase creation complete.")
