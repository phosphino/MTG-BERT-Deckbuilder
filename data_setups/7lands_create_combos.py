import os
import re
import numpy as np
import torch
from scipy.sparse import coo_matrix
from sqlalchemy import create_engine, MetaData, select

print("Starting...")

# ---------------------------------------------------------
# Paths & DB Setup
# ---------------------------------------------------------
games_folder = r"C:\Users\breuh\OneDrive\proggy\python\MTG\roberta\data_setups\training_database.db"
games_folder = os.path.normpath(games_folder)

engine = create_engine("sqlite:///" + games_folder)
metadata = MetaData()
metadata.reflect(bind=engine)

# Match tablenames like "SET_FORMAT" (e.g., "DMU_BestOfOne")
gamedata_tablename_pattern = re.compile(r"([A-Z]{3})_(\D+)")
gamedata_tablenames = list(filter(gamedata_tablename_pattern.match, metadata.tables.keys()))

# ---------------------------------------------------------
# Filter out basic lands from Scryfall metadata
# ---------------------------------------------------------
scryfall_table = metadata.tables["scryfall_pruned"]

query = select(scryfall_table.c.name).where(scryfall_table.c.type_line.ilike("%basic land%"))
with engine.connect() as conn:
    land_names = conn.execute(query).fetchall()
land_names = [x[0] for x in land_names]

print(f"Excluded basic lands: {land_names}")


# ---------------------------------------------------------
# Helper: save sparse matrices + metadata
# ---------------------------------------------------------
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
            "P": "Win rate when card i and j are played together (MW / MM)",
            "cardnames": "Index-aligned list of card names, so cardnames[i] maps to row/col index i",
            "dtype": "All matrices stored as scipy.sparse.coo_matrix",
            "note": "Diagonals represent per-card stats; off-diagonals are pairwise combos.",
            "n_cards": len(cardnames),
            "n_games": row_base,
            "source_db": games_folder,
        },
    }

    torch.save(data, path)
    print(f"Saved combo graph to {path}")


# ---------------------------------------------------------
# Helper: get all non-land card columns for a given table
# ---------------------------------------------------------
def table_card_columns(table, land_names):
    return [
        c.key
        for c in table.columns
        if c.key not in land_names and c.key not in ("won", "index")
    ]


# ---------------------------------------------------------
# 1) Build global card vocabulary across all tables
# ---------------------------------------------------------
all_cards = set()
for tablename in gamedata_tablenames:
    table = metadata.tables[tablename]
    cols = table_card_columns(table, land_names)
    all_cards.update(cols)

# Stable ordering for reproducibility
global_all_cards = sorted(all_cards)

# Card name → global index
ctoi = {c: i for i, c in enumerate(global_all_cards)}
ncols = len(global_all_cards)


# ---------------------------------------------------------
# 2) Stream tables into sparse COO representation
# ---------------------------------------------------------
rows_idx, cols_idx, data_vals = [], [], []  # COO components
y_parts = []                                # outcome labels
row_base = 0                                # global row offset
CHUNK = 50_000                              # DB fetch size

for tablename in gamedata_tablenames:
    print(f"Processing {tablename}...")
    table = metadata.tables[tablename]
    cols = table_card_columns(table, land_names)
    itoi_local_to_global = np.array([ctoi[c] for c in cols], dtype=int)

    # SELECT card cols + outcome col
    query = [table.c[c] for c in cols] + [table.c.won]
    query = select(*query)

    # Stream from DB in chunks
    with engine.connect().execution_options(stream_results=True) as conn:
        result = conn.execute(query)
        while True:
            chunk = result.fetchmany(CHUNK)
            if not chunk:
                break

            chunk = list(chunk)
            X = np.array([c[:-1] for c in chunk], dtype=int)  # card matrix
            Y = np.array([c[-1] for c in chunk], dtype=int)   # outcome labels

            y_parts.append(Y)

            # Sparse nonzeros
            row_local, col_local = np.nonzero(X)
            if len(row_local):
                rows_idx.append(row_base + row_local)
                cols_idx.append(itoi_local_to_global[col_local])
                data_vals.append(np.ones_like(col_local, dtype=int))

            row_base += X.shape[0]


# ---------------------------------------------------------
# 3) Build sparse matrices
# ---------------------------------------------------------
rows_idx = np.concatenate(rows_idx)
cols_idx = np.concatenate(cols_idx)
data_vals = np.concatenate(data_vals)

y = np.concatenate(y_parts).astype(int)

# Game-card presence matrix (rows = games, cols = cards)
M = coo_matrix((data_vals, (rows_idx, cols_idx)),
               shape=(row_base, ncols), dtype=int).tocsr()

# Card-card co-occurrence (games played together)
MM = (M.T @ M).tocoo()

# Card-card wins (wins played together)
MW = (M.T @ (M.multiply(y.reshape(-1, 1)))).tocoo()

# Avoid division by zero when computing win rates
MM_reciprocal = MM.copy()
MM_reciprocal.data = np.divide(
    1, MM_reciprocal.data,
    out=np.zeros_like(MM_reciprocal.data, dtype=float),
    where=MM_reciprocal.data != 0
)

# Card-card win rates
P = MM_reciprocal.multiply(MW).tocoo()


# ---------------------------------------------------------
# 4) Save results
# ---------------------------------------------------------
save_sparse_combo_graph(MM, MW, P, global_all_cards, "combo_matrices.pt")
