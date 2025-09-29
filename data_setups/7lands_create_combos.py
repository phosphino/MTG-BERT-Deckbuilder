"""
Processes Magic: The Gathering game data from a 17lands SQLite database to
calculate pairwise card performance metrics.

This script performs the following steps:
1. Connects to the database and identifies all game data tables.
2. Builds a global vocabulary of all unique, non-basic-land card names.
3. Streams game data in chunks to efficiently handle large datasets.
4. Converts the streamed data into a single large sparse matrix `M` (games x cards)
   and a vector `y` (game outcomes).
5. Calculates three key matrices using sparse algebra:
   - MM: Card-card co-occurrence (how often pairs appear together).
   - MW: Card-card co-wins (how often pairs win together).
   - P: Pairwise win rate (MW / MM).
6. Saves these computed matrices to a PyTorch file ('combo_matrices.pt').
"""
import os
import re

import numpy as np
import torch
from scipy.sparse import coo_matrix
from sqlalchemy import create_engine, MetaData, Table, select

print('Starting')

excluded_cards = []

# --- 1. Configuration & Database Setup ---

# Define the path to the SQLite database file.
GAMES_DB_PATH = os.path.normpath(r"C:\Users\breuh\OneDrive\proggy\python\MTG\roberta\data_setups\training_database.db")

# Define a regular expression to identify tables containing game data.
# These tables typically follow a 'SET_FORMAT' naming convention (e.g., 'LCI_PremierDraft').
GAMEDATA_TABLENAME_PATTERN = re.compile(r"([A-Z]{3})_(\D+)")

# Set the database connection using SQLAlchemy.
engine = create_engine("sqlite:///" + GAMES_DB_PATH)
metadata = MetaData()
metadata.reflect(bind=engine) # Inspect the database to load table schemas.

# Filter all table names to get only the ones matching our game data pattern.
gamedata_tablenames = list(filter(GAMEDATA_TABLENAME_PATTERN.match, metadata.tables.keys()))
scryfall_table = metadata.tables['scryfall_pruned']

# Query the database to get a list of all basic land names.
# These will be excluded from all performance calculations.
query = select(scryfall_table.c.name).where(scryfall_table.c.type_line.ilike('%basic land%'))
with engine.connect() as conn:
    land_names = conn.execute(query).fetchall()
land_names = [x[0] for x in land_names]
print(f"Found {len(land_names)} basic lands.")
print("\t",land_names)

# --- 2. Helper Functions ---

def save_sparse_combo_graph(
    MM: coo_matrix,
    MW: coo_matrix,
    P: coo_matrix,
    cardnames: list[str],
    path: str
):
    """
    Saves card co-occurrence, co-win, and win-rate matrices into a single .pt file.

    Args:
        MM (coo_matrix): Card-card co-occurrence counts (# games played together).
        MW (coo_matrix): Card-card win counts (# wins together).
        P (coo_matrix): Card-card win rates (MW / MM).
        cardnames (list[str]): Ordered list of card names, index-aligned with the matrices.
        path (str): File path to save the torch file.
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


def table_card_columns(table: Table, cards_excluded_names: list[str]) -> list[str]:
    """
    Returns all relevant card-related column names from a SQLAlchemy Table object.

    Excludes:
    - Basic lands (provided in `cards_excluded_names`).
    - The 'won' outcome column.
    - Any stray 'index' columns.
    """
    cols = [
        c.key for c in table.columns
        if c.key not in cards_excluded_names and c.key not in ('won', 'index')
    ]
    return cols


# --- 3. Build Global Card Vocabulary ---
# To ensure matrix integrity, we first create a single, master list of all
# unique cards across all game tables. This establishes a stable, global index for each card.

print("Building global card vocabulary...")
# excluded_cards = land_names
print("\tExcluding cards:", excluded_cards)
all_cards = set()
for tablename in gamedata_tablenames:
    table = metadata.tables[tablename]
    cols = table_card_columns(table, excluded_cards)
    all_cards.update(cols) # Add this table's cards to the global set.

# A sorted list provides a consistent, deterministic order.
global_all_cards = sorted(list(all_cards))
# The card-to-index (ctoi) map allows for O(1) lookups of a card's global index.
ctoi = {c: i for i, c in enumerate(global_all_cards)}
ncols = len(global_all_cards)
print(f"Vocabulary created with {ncols} unique cards.")


# --- 4. Stream and Process Game Data into a Sparse Matrix ---
# This is the core processing loop. We build a sparse matrix `M` where each row is a
# game and each column is a card. `M[i, j] = 1` if card `j` was in game `i`.
# Data is streamed in chunks to avoid loading the entire dataset into memory.

print("Streaming and processing game data from all tables...")

# Components for building a COO (coordinate list) sparse matrix.
# This format is efficient for incremental construction.
rows_idx, cols_idx, data_vals = [], [], []
# List to store the outcome (win/loss) for each game, in order.
y_parts = []
# A global row offset to ensure that games from different tables get unique row indices.
row_base = 0

# Configuration for data streaming.
CHUNK_SIZE = 50_000
PARANOID_SAMPLE_COLS = 5 # Number of columns to sample for the sanity check.

for tablename in gamedata_tablenames:
    print(f"  Processing {tablename}...")
    table = metadata.tables[tablename]

    # Get the list of card columns present in this specific table.
    local_cols = table_card_columns(table, excluded_cards)

    # Sanity check: ensure all cards in this table exist in our global vocabulary.
    missing = [c for c in local_cols if c not in ctoi]
    if missing:
        raise KeyError(f"{tablename}: {len(missing)} local columns not in global vocab: {missing[:5]}...")

    # Construct the SQL query. While a specific order is requested, the script's logic
    # correctly handles any reordering by the database by checking the result's keys.
    select_cols = [table.c[c] for c in local_cols] + [table.c.won]
    query = select(*select_cols)

    # Stream from the database in chunks.
    with engine.connect().execution_options(stream_results=True) as conn:
        result = conn.execute(query)

        # CRITICAL: Do not trust the DB to return columns in the requested order.
        # Instead, get the *actual* column order from the result cursor. This is the
        # authoritative order for this table.
        result_keys = list(result.keys())
        if "won" not in result_keys:
            raise RuntimeError(f"{tablename}: 'won' column missing from result set.")

        # Determine the index of the 'won' column and the ordered list of card columns.
        won_idx = result_keys.index("won")
        ordered_card_cols = [k for i, k in enumerate(result_keys) if i != won_idx]

        # Sanity check: ensure the columns we received match what we expected.
        if set(ordered_card_cols) != set(local_cols):
            raise RuntimeError(f"{tablename}: Mismatch between requested local_cols and result columns.")

        # Build the local-to-global index map based on the *actual* result order.
        # This map translates this table's local column indices to the master global indices.
        itoi_local_to_global = np.fromiter((ctoi[c] for c in ordered_card_cols), dtype=np.int16)

        while True:
            chunk = result.fetchmany(CHUNK_SIZE)
            if not chunk:
                break # End of data for this table.

            arr = np.asarray(chunk, dtype=np.int8)

            # Split the numpy array into features (X) and outcomes (Y).
            Y = arr[:, won_idx]
            X = np.delete(arr, won_idx, axis=1)

            y_parts.append(Y)

            # Find the coordinates of all non-zero elements (i.e., where a card is present).
            row_local, col_local = np.nonzero(X)
            if row_local.size:
                # Append the coordinates to our global COO lists.
                # Adjust local row indices by the global `row_base` offset.
                rows_idx.append((row_base + row_local).astype(np.int64, copy=False))
                # Map local column indices to their global equivalents.
                cols_idx.append(itoi_local_to_global[col_local].astype(np.int16, copy=False))
                # All presence values are 1.
                data_vals.append(np.ones_like(col_local, dtype=np.int8))

                # --- Paranoid runtime sanity check ---
                # This check confirms that our index mapping is correct, preventing
                # silent data corruption if column orders were unexpectedly shifted.
                sample_indices = np.unique(col_local)[:PARANOID_SAMPLE_COLS]
                for cl in sample_indices:
                    gidx = int(itoi_local_to_global[cl])
                    local_name = ordered_card_cols[int(cl)]
                    global_name = global_all_cards[gidx]
                    if local_name != global_name:
                        raise AssertionError(
                            f"{tablename}: Column mapping mismatch! local[{cl}]='{local_name}' "
                            f"!= global[{gidx}]='{global_name}'"
                        )

            # Increment the global row offset for the next chunk/table.
            row_base += X.shape[0]


# --- 5. Final Matrix Construction & Calculation ---

print("Finalizing matrices...")
# Concatenate all the chunked COO components into single numpy arrays.
rows_idx = np.concatenate(rows_idx)
cols_idx = np.concatenate(cols_idx)
data_vals = np.concatenate(data_vals)

# Concatenate outcome labels and cast to integer.
y = np.concatenate(y_parts).astype(np.int32)

# Assemble the final sparse matrix `M` in CSR (Compressed Sparse Row) format,
# which is efficient for matrix multiplication.
M = coo_matrix((data_vals, (rows_idx, cols_idx)),
               shape=(row_base, ncols), dtype=np.int32).tocsr()

# --- Pairwise Metric Calculations using Sparse Matrix Algebra ---

# MM = M.T @ M: The card-card co-occurrence matrix.
# The element MM[i, j] is the total number of games where card `i` and `j`
# appeared together in the same deck.
MM = (M.T @ M).tocoo()

# MW = M.T @ (M with losing games zeroed out): The card-card co-win matrix.
# M.multiply(y.reshape(-1, 1)) zeroes out all rows for games that were lost (y=0).
# The subsequent multiplication then counts only the wins.
# The element MW[i, j] is the total number of *wins* where card `i` and `j`
# appeared together.
MW = M.T @ (M.multiply(y.reshape(-1, 1))).tocoo()

# P = MW / MM: The pairwise win rate matrix.
# This performs safe element-wise division to get the win rate for each pair.
MM_reciprocal = MM.copy()
# Safely compute 1/x, returning 0 where the denominator is 0 to avoid errors.
MM_reciprocal.data = np.divide(
    1, MM_reciprocal.data,
    out=np.zeros_like(MM_reciprocal.data, dtype=float),
    where=MM_reciprocal.data != 0
)
P = MM_reciprocal.multiply(MW).tocoo()

# text = f""

# Save the final computed matrices and the card name vocabulary.
save_sparse_combo_graph(MM, MW, P, global_all_cards, "combo_matrices.pt")

print("Processing complete.")