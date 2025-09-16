## ==========================================================
## Script: scryfall_prune_to_db.py
## Author: Andy Breuhaus
##
## Description:
##   - Load latest Scryfall Oracle JSON dump
##   - Normalize text fields (remove accents, clean encoding)
##   - Keep only selected card attributes
##   - Filter:
##       * English-language cards
##       * Cards legal in at least one supported format
##       * Remove Arena rebalanced alternates ("A-" prefixed)
##       * Drop rows with <UNK> placeholders
##   - Store cleaned dataset into SQLite for downstream use
##
## Inputs:
##   - Scryfall Oracle JSON file: project_root/scryfall_json/oracle-cards-*.json
##
## Outputs:
##   - SQLite database file: training_database.db
##   - Table: scryfall_pruned
##
## Database Schema:
##   - One row per card
##   - Columns: mana_cost, type_line, power, toughness,
##              oracle_text, colors, name, cmc,
##              color_identity, keywords, set
##
## Usage:
##   Run this script directly:
##       python scryfall_prune_to_db.py
## ==========================================================

import os
import re
import json
import polars as pl
from polars import selectors as cs
from pathlib import Path
from sqlalchemy import create_engine

from utils import remove_accents


# ----------------------------------------------------------
# Config: Features and Filters
# ----------------------------------------------------------
scryfall_legalities = [
    "standard", "pioneer", "modern", "explorer",
    "historic", "vintage", "legacy"
]

scryfall_feature_columns = [
    "mana_cost", "type_line", "power", "toughness",
    "oracle_text", "colors", "name", "cmc",
    "color_identity", "keywords", "set"
]

filter_columns = ["lang", "legalities"]


# ----------------------------------------------------------
# Project Paths
# ----------------------------------------------------------
project_root = Path(r"C:/Users/breuh/OneDrive/proggy/python/MTG/roberta")

scryfall_folder = project_root / "scryfall_json"
oracle_pattern = re.compile(r"oracle-cards-(\d{8})\d{6}\.json")
oracle_paths = sorted(
    [p for p in scryfall_folder.iterdir() if oracle_pattern.search(p.name)],
    key=lambda x: x.name,
    reverse=True
)
scryfall_path = oracle_paths[0]  # Latest Oracle file

export_path = os.path.normpath(
    r"C:\Users\breuh\OneDrive\proggy\python\MTG\roberta\data_setups\training_database.db"
)


# ----------------------------------------------------------
# Load Scryfall JSON
# ----------------------------------------------------------
with open(scryfall_path, "r", encoding="utf-8") as f:
    scryfall_txt = json.load(f)

# Normalize accents in selected text fields
for index, card in enumerate(scryfall_txt):
    new_card = {}
    for k, v in card.items():
        if k in ("name", "oracle_text"):
            new_card[k] = remove_accents(v)
        else:
            new_card[k] = v
    scryfall_txt[index] = new_card


# ----------------------------------------------------------
# Subset fields of interest
# ----------------------------------------------------------
scryfall_txt_subset = [
    {k: v for k, v in card.items() if k in scryfall_feature_columns + filter_columns}
    for card in scryfall_txt
]

df_utf8 = pl.from_dicts(scryfall_txt_subset)


# ----------------------------------------------------------
# Filtering
# ----------------------------------------------------------
# Legalities: only keep formats actually present in schema
available_legality_fields = [
    l for l in scryfall_legalities
    if l in [fld.name for fld in df_utf8.schema["legalities"].fields]
]
legality_filter = [
    pl.col("legalities").struct.field(fmt) == "legal"
    for fmt in available_legality_fields
]

# Apply legality + language filter
df_utf8 = df_utf8.filter(pl.any_horizontal(legality_filter) & (pl.col("lang") == "en"))

# Keep only chosen features
df_utf8 = (
    df_utf8
    .select(scryfall_feature_columns)
    .filter(~pl.col("name").str.starts_with("A-"))  # drop Arena alternates
    .sort("name")
)

# Convert list columns → comma-separated strings
df_utf8 = df_utf8.with_columns(cs.by_dtype(pl.List(str)).list.join(", "))

# Drop rows containing "<UNK>" placeholders
before = len(df_utf8)
df_for_export = df_utf8.filter(
    ~pl.any_horizontal([pl.col(c) == "<UNK>" for c in ["name", "oracle_text"]])
)
after = len(df_for_export)
print(f"Dropped {before - after} rows due to <UNK> values.")


# ----------------------------------------------------------
# Export to SQLite
# ----------------------------------------------------------
engine_path = "sqlite:///" + export_path
engine = create_engine(engine_path)

df_for_export.write_database(
    table_name="scryfall_pruned",
    connection=engine_path,
    if_table_exists="replace",
    engine="adbc"
)

# ----------------------------------------------------------
# Indexing & Optimization
# ----------------------------------------------------------
with engine.connect() as conn:
    # A) Index on name: speeds up exact and prefix queries
    conn.exec_driver_sql("""
        CREATE INDEX IF NOT EXISTS ix_scryfall_name
        ON scryfall_pruned(name);
    """)

    # Gather stats & optimize query planning
    conn.exec_driver_sql("ANALYZE;")
    conn.exec_driver_sql("PRAGMA optimize;")
    conn.commit()

print("\nScryfall pruning and export complete.")
