"""
Processes Magic: The Gathering game data from a 17Lands-derived SQLite database
to calculate pairwise card performance metrics.

This script performs the following steps:
1. Connects to the SQLite database and identifies all game data tables.
2. Builds a global vocabulary of all unique non-basic-land card names.
3. Streams game data in chunks to efficiently handle large datasets.
4. Constructs a large sparse matrix `M` (games × cards) where rows are games and
   columns are cards, along with an outcome vector `y` (win/loss).
5. Uses sparse matrix algebra to compute:
   - MM: Card–card co-occurrence counts (# of games cards i and j were played together).
   - MW: Card–card co-win counts (# of wins where i and j were played together).
   - P:  Pairwise win rate matrix (MW / MM, element-wise).
6. Saves these computed matrices, along with metadata, to a PyTorch file
   (`combo_matrices.pt`) in a robust, deconstructed format.
"""
import os
import re

import numpy as np
import torch
from scipy.sparse import coo_matrix, csr_matrix
from sqlalchemy import create_engine, MetaData, Table, select

print('Starting')

# --- 1. Configuration & Database Setup ---

GAMES_DB_PATH = os.path.normpath(r"C:\Users\breuh\OneDrive\proggy\python\MTG\roberta\data_setups\training_database.db")
GAMEDATA_TABLENAME_PATTERN = re.compile(r"([A-Z]{3})_(\D+)")

engine = create_engine("sqlite:///" + GAMES_DB_PATH)
metadata = MetaData()
metadata.reflect(bind=engine)

gamedata_tablenames = list(filter(GAMEDATA_TABLENAME_PATTERN.match, metadata.tables.keys()))
scryfall_table = metadata.tables['scryfall_pruned']

query = select(scryfall_table.c.name).where(scryfall_table.c.type_line.ilike('%basic land%'))
with engine.connect() as conn:
    land_names = conn.execute(query).fetchall()
land_names = [x[0] for x in land_names]
print(f"Found {len(land_names)} basic lands.")


# --- 2. Helper Functions ---

def save_combo_data_robust(
    M: csr_matrix,
    y: np.ndarray,
    MM: coo_matrix,
    MW: coo_matrix,
    P: coo_matrix,
    cardnames: list[str],
    path: str
):
    """
    Saves sparse matrices by deconstructing them into tensors for robust serialization.
    This prevents data corruption issues that can occur when saving complex SciPy
    objects directly with torch.save().
    """
    def matrix_to_tensor_dict(matrix: coo_matrix) -> dict:
        # Converts a SciPy COO matrix to a dictionary of PyTorch tensors.
        if not isinstance(matrix, coo_matrix):
            matrix = matrix.tocoo()
        return {
            "data": torch.from_numpy(matrix.data),
            "row": torch.from_numpy(matrix.row),
            "col": torch.from_numpy(matrix.col),
            "shape": matrix.shape
        }

    data_to_save = {
        "M": matrix_to_tensor_dict(M),
        "y": torch.from_numpy(y),
        "MM": matrix_to_tensor_dict(MM),
        "MW": matrix_to_tensor_dict(MW),
        "P": matrix_to_tensor_dict(P),
        "cardnames": cardnames,
        "meta": {
        "description": (
            "Pairwise card performance data derived from 17Lands game tables. "
            "Matrices were constructed by streaming SQLite tables in chunks, "
            "excluding basic lands from the vocabulary, and mapping all card "
            "names to a global index."
        ),
        "M": (
            "Game–card presence matrix (csr_matrix). "
            "Shape = (n_games, n_cards). Entry M[i,j] = 1 if card j was played in game i."
        ),
        "y": (
            "Outcome vector (numpy.ndarray of int8). "
            "Length = n_games. Entry y[i] = 1 if game i was won, 0 otherwise."
        ),
        "MM": (
            "Card–card co-occurrence matrix (coo_matrix). "
            "Shape = (n_cards, n_cards). Entry MM[i,j] = number of games in which cards i and j were both played."
        ),
        "MW": (
            "Card–card co-win matrix (coo_matrix). "
            "Shape = (n_cards, n_cards). Entry MW[i,j] = number of wins where cards i and j were both played."
        ),
        "P": (
            "Pairwise win rate matrix (coo_matrix). "
            "Shape = (n_cards, n_cards). Entry P[i,j] = MW[i,j] / MM[i,j], "
            "the empirical win rate when i and j appear together. Division by zero handled with 0."
        ),
        "cardnames": (
            "Index-aligned list of card names. "
            "cardnames[k] gives the card corresponding to column k in all matrices."
        ),
        "dimensions": {
            "n_games": row_base,
            "n_cards": len(cardnames),
        },
        "dtype_notes": {
            "M": "int32 values, binary presence (0 or 1).",
            "y": "int8 values (0 or 1).",
            "MM": "int32 counts of games played together.",
            "MW": "int32 counts of wins together.",
            "P": "float win rates, element-wise division MW / MM."
        },
        "processing_notes": {
            "chunk_size": CHUNK_SIZE,
            "paranoid_column_check": f"Verified first {PARANOID_SAMPLE_COLS} local→global mappings per chunk.",
            "exclusions": f"{len(land_names)} basic lands excluded from vocabulary.",
            "source_db": GAMES_DB_PATH
        }
    }
    }
    torch.save(data_to_save, path)
    print(f"Saved combo data components to {path}")


def load_combo_data_robust(path: str) -> dict:
    """
    Loads sparse matrix components from a .pt file and reconstructs the
    SciPy sparse matrix objects.
    """
    loaded_data = torch.load(path, weights_only=False)

    def tensor_dict_to_matrix(tensor_dict: dict, format: str = 'coo') -> coo_matrix:
        # Reconstructs a SciPy sparse matrix from a dictionary of tensors.
        matrix = coo_matrix(
            (tensor_dict["data"].numpy(), (tensor_dict["row"].numpy(), tensor_dict["col"].numpy())),
            shape=tensor_dict["shape"]
        )
        if format == 'csr':
            return matrix.tocsr()
        return matrix

    reconstructed_data = {
        "M": tensor_dict_to_matrix(loaded_data["M"], format='csr'),
        "y": loaded_data["y"].numpy(),
        "MM": tensor_dict_to_matrix(loaded_data["MM"]),
        "MW": tensor_dict_to_matrix(loaded_data["MW"]),
        "P": tensor_dict_to_matrix(loaded_data["P"]),
        "cardnames": loaded_data["cardnames"],
        "meta": loaded_data.get("meta", {})
    }
    print(f"Successfully loaded and reconstructed data from {path}")
    return reconstructed_data


def table_card_columns(table: Table, cards_excluded_names: list[str] = []) -> list[str]:
    """Extracts card-related columns from a game table, excluding unwanted ones."""
    return [
        c.key for c in table.columns
        if c.key not in cards_excluded_names and c.key not in ('won', 'index')
    ]

if __name__ == "__main__":

    # --- 3. Build Global Card Vocabulary ---

    print("Building global card vocabulary...")
    excluded_cards = land_names  # FIX: Use the fetched land names for exclusion.
    excluded_cards = []
    print(f"\tExcluding {len(excluded_cards)} cards (e.g., {excluded_cards[:5]}...).")

    all_cards = set()
    for tablename in gamedata_tablenames:
        table = metadata.tables[tablename]
        cols = table_card_columns(table, excluded_cards)
        all_cards.update(cols)

    global_all_cards = sorted(list(all_cards))
    ctoi = {c: i for i, c in enumerate(global_all_cards)}
    ncols = len(global_all_cards)
    print(f"Vocabulary created with {ncols} unique cards.")


    # --- 4. Stream and Process Game Data into a Sparse Matrix ---

    print("Streaming and processing game data from all tables...")

    rows_idx, cols_idx, data_vals = [], [], []
    y_parts = []
    row_base = 0

    CHUNK_SIZE = 50_000
    PARANOID_SAMPLE_COLS = 5

    for tablename in gamedata_tablenames:
        print(f"  Processing {tablename}...")
        table = metadata.tables[tablename]
        local_cols = table_card_columns(table, excluded_cards)

        missing = [c for c in local_cols if c not in ctoi]
        if missing:
            raise KeyError(f"{tablename}: {len(missing)} columns missing from global vocab: {missing[:5]}...")

        select_cols = [table.c[c] for c in local_cols] + [table.c.won]
        query = select(*select_cols)

        with engine.connect().execution_options(stream_results=True) as conn:
            result = conn.execute(query)
            result_keys = list(result.keys())
            if "won" not in result_keys:
                raise RuntimeError(f"{tablename}: 'won' column missing from result set.")

            won_idx = result_keys.index("won")
            ordered_card_cols = [k for i, k in enumerate(result_keys) if i != won_idx]

            if set(ordered_card_cols) != set(local_cols):
                raise RuntimeError(f"{tablename}: Column mismatch between schema and result set.")
            
            # FIX: Use np.int32 for indices to prevent overflow if ncols > 32,767.
            itoi_local_to_global = np.fromiter((ctoi[c] for c in ordered_card_cols), dtype=np.int32)

            while True:
                chunk = result.fetchmany(CHUNK_SIZE)
                if not chunk:
                    break

                arr = np.asarray(chunk, dtype=np.int8)
                Y = arr[:, won_idx]
                X = np.delete(arr, won_idx, axis=1)

                y_parts.append(Y)
                row_local, col_local = np.nonzero(X)
                if row_local.size:
                    # FIX: Use np.int32 for consistency and safety.
                    rows_idx.append((row_base + row_local).astype(np.int32, copy=False))
                    cols_idx.append(itoi_local_to_global[col_local].astype(np.int32, copy=False))
                    data_vals.append(np.ones_like(col_local, dtype=np.int8))

                    sample_indices = np.unique(col_local)[:PARANOID_SAMPLE_COLS]
                    for cl in sample_indices:
                        gidx = int(itoi_local_to_global[cl])
                        local_name = ordered_card_cols[int(cl)]
                        global_name = global_all_cards[gidx]
                        if local_name != global_name:
                            raise AssertionError(
                                f"{tablename}: Column mapping mismatch! "
                                f"local[{cl}]='{local_name}' != global[{gidx}]='{global_name}'"
                            )
                row_base += X.shape[0]


    # --- 5. Final Matrix Construction & Pairwise Metrics ---

    print("Finalizing matrices...")
    rows_idx = np.concatenate(rows_idx)
    cols_idx = np.concatenate(cols_idx)
    data_vals = np.concatenate(data_vals)

    # FIX: np.int8 is sufficient for win/loss (0 or 1).
    y = np.concatenate(y_parts).astype(np.int8)

    # FIX: Use np.int32 to prevent overflow during matrix multiplication.
    M = coo_matrix((data_vals, (rows_idx, cols_idx)),
                shape=(row_base, ncols), dtype=np.int32).tocsr()

    MM = (M.T @ M).tocoo()
    MW = (M.T @ M.multiply(y.reshape(-1, 1))).tocoo()

    MM_reciprocal = MM.copy()
    MM_reciprocal.data = np.divide(
        1, MM_reciprocal.data,
        out=np.zeros_like(MM_reciprocal.data, dtype=float),
        where=MM_reciprocal.data != 0
    )
    P = MM_reciprocal.multiply(MW).tocoo()

    # FIX: Call the new, robust save function.
    save_combo_data_robust(M, y, MM, MW, P, global_all_cards, "combo_matrices.pt")

    print("Processing complete.")