import os, sys, re
import json
import numpy as np
import torch 
from torch.utils.data import Dataset, DataLoader
import polars as pl
from polars import selectors as cs
import pyarrow.feather as feather
from functools import partial
from datasets import Dataset as DSet
from datasets import DatasetDict as DDict
from datasets import concatenate_datasets

import pandas as pd

from scipy.sparse import csr_matrix, coo_matrix

from transformers import AutoTokenizer, AutoModel, AutoModelForMaskedLM, Trainer, TrainingArguments, TextDataset, RobertaForMaskedLM

from sqlalchemy import create_engine, MetaData, Table, select, func
import random

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from itertools import compress, chain
from collections import defaultdict

print('Starting')

games_folder = r"C:\Users\breuh\OneDrive\proggy\python\MTG\roberta\data_setups\training_database.db"
games_folder = os.path.normpath(games_folder)

engine = create_engine("sqlite:///"+games_folder)
metadata = MetaData()
metadata.reflect(bind=engine)
gamedata_tablename_pattern = re.compile(r"([A-Z]{3})_(\D+)")
gamedata_tablenames = list(filter(gamedata_tablename_pattern.match, metadata.tables.keys()))

scryfall_table = metadata.tables['scryfall_pruned']

query = select(scryfall_table.c.name).where(scryfall_table.c.type_line.ilike('%basic land%'))
with engine.connect() as conn:
    land_names = conn.execute(query).fetchall()
land_names = [x[0] for x in land_names]
print(land_names)

def save_sparse_combo_graph(MM: coo_matrix,
                            MW: coo_matrix,
                            P: coo_matrix,
                            cardnames: list[str],
                            path: str):
    """
    Save card co-occurrence and win matrices into a single torch file.

    Args:
        MM (coo_matrix): card-card co-occurrence counts (# games played together)
        MW (coo_matrix): card-card win counts (# wins together)
        P  (coo_matrix): card-card win rates (MW / MM)
        cardnames (list[str]): ordered list of card names, index-aligned
        path (str): file path to save (.pt)
    """

    data = {
        "MM": MM,             # scipy.sparse.coo_matrix
        "MW": MW,             # scipy.sparse.coo_matrix
        "P": P,               # scipy.sparse.coo_matrix
        "cardnames": cardnames,  # list of str
        "meta": {
            "MM": "Number of games card i and card j were played together",
            "MW": "Number of wins where card i and card j were played together",
            "P":  "Win rate when card i and j are played together (MW / MM)",
            "cardnames": "Index-aligned list of card names, so cardnames[i] maps to row/col index i",
            "dtype": "All matrices stored as scipy.sparse.coo_matrix",
            "note": "Diagonals represent per-card stats; off-diagonals are pairwise combos."
        }
    }

    torch.save(data, path)
    print(f"Saved combo graph to {path}")

def table_card_columns(table, land_names):
    # Return all card-related columns for a table, excluding:
    # - any basic lands (provided as land_names)
    # - the 'won' outcome column
    # - any stray 'index' column
    cols = [c.key for c in table.columns if c.key not in land_names and c.key not in ('won', 'index')]
    return cols

# --- 1) Build global card vocabulary across all tables ---
all_cards = set()
for tablename in gamedata_tablenames:
    table = metadata.tables[tablename]
    cols = table_card_columns(table, land_names)  # get card columns for this table
    all_cards.update(cols)                        # accumulate into union of all cards

# Sorted global list of all card names (stable order for consistent indexing)
global_all_cards = sorted(all_cards)

# Map card name → global column index
ctoi = {c:i for i,c in enumerate(global_all_cards)}
ncols = len(global_all_cards)

# --- 2) Stream tables into sparse COO representation ---
# COO components (row indices, col indices, values)
rows_idx, cols_idx, data_vals = [], [], []
# Labels and outcomes
y_parts, set_labels, format_labels = [], [], []
# Track row offset as we append games from multiple tables
row_base = 0
# How many rows to pull per fetch from SQL
CHUNK = 50_000  # tune for your available RAM

for tablename in gamedata_tablenames:
    print(tablename)
    table = metadata.tables[tablename]
    # Local card columns for this table
    cols = table_card_columns(table, land_names)
    # Map local column indices → global column indices
    itoi_local_to_global = np.array([ctoi[c] for c in cols], dtype=int)
    # Build SELECT query: all local card columns + outcome column 'won'
    query = [table.c[c] for c in cols] + [table.c.won]
    query = select(*query)

    # Stream results from DB (no full materialization in RAM)
    with engine.connect().execution_options(stream_results=True) as conn:
        result = conn.execute(query)
        while True:
            # Fetch up to CHUNK rows from DB
            chunk = result.fetchmany(CHUNK)
            if not chunk:
                break
            # Materialize chunk into list of tuples
            chunk = list(chunk)

            # Split into X (card counts per game) and Y (win labels)
            X = np.array([c[:-1] for c in chunk], dtype=int)  # all but last element
            Y = np.array([c[-1] for c in chunk], dtype=int)   # last element = 'won'

            # Collect labels
            y_parts.append(Y)

            # Find all nonzero entries in this chunk
            row_local, col_local = np.nonzero(X)
            if len(row_local):
                # Append row indices adjusted by row_base (global offset)
                rows_idx.append(row_base + row_local)
                # Map local column indices → global column indices
                cols_idx.append(itoi_local_to_global[col_local])
                # All nonzeros have value 1 (binarized presence)
                data_vals.append(np.ones_like(col_local, dtype=int))
                
            # Increment row offset by number of games in this chunk
            row_base += X.shape[0]

# --- 3) Concatenate all collected COO components ---
rows_idx = np.concatenate(rows_idx)
cols_idx = np.concatenate(cols_idx)
data_vals = np.concatenate(data_vals)

# Concatenate outcome labels
y = np.concatenate(y_parts)
y = y.astype(int)

M = coo_matrix((data_vals, (rows_idx, cols_idx)),
               shape=(row_base, ncols), dtype=int).tocsr()

MM = (M.T@M).tocoo()
MW = M.T@(M.multiply(y.reshape(-1, 1))).tocoo()
MM_reciprocal = MM.copy()
MM_reciprocal.data = np.divide(
    1, MM_reciprocal.data,
    out=np.zeros_like(MM_reciprocal.data, dtype=float),
    where=MM_reciprocal.data != 0
)
P = MM_reciprocal.multiply(MW).tocoo()

save_sparse_combo_graph(MM, MW, P, global_all_cards, "combo_matrices.pt")
