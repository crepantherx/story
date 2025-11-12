# api.py
from fastapi import FastAPI, Depends, HTTPException, status, Request
from typing import List, Optional, Dict, Tuple
import sqlite3
import threading
import os
import re
import json

import numpy as np
import pandas as pd
from annoy import AnnoyIndex

app = FastAPI()

# Path to your sqlite DB (change if needed)
DB_PATH = "/Users/sudhirsingh/PyCharmProjects/story/src/services/frontend/data.db"

# Annoy index globals + lock for thread-safety
INDEX_LOCK = threading.Lock()
ANNOY_INDEX: Optional[AnnoyIndex] = None
ID_MAP: Dict[int, str] = {}
VECTOR_DIM: Optional[int] = None

# Auth token (kept simple to match your current code)
EXPECTED_TOKEN = "sudhir@123"


# -------------------------
# Utilities: sqlite + vectors
# -------------------------
def get_conn(db_path: str = DB_PATH):
    if not os.path.exists(db_path):
        raise RuntimeError(f"DB file not found at: {db_path}")
    conn = sqlite3.connect(db_path, timeout=10.0, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def str_to_array(s: Optional[str]) -> Optional[np.ndarray]:
    """Parse embedding column string to numpy array."""
    if s is None:
        return None
    if isinstance(s, (list, np.ndarray)):
        return np.array(s, dtype=float)
    t = str(s).strip()
    if t.startswith("[") and t.endswith("]"):
        t = t[1:-1]
    t = re.sub(r"[,\s]+", " ", t.strip())
    if t == "":
        return None
    try:
        arr = np.fromstring(t, sep=" ", dtype=float)
        if arr.size == 0:
            return None
        return arr
    except Exception:
        return None


def array_to_str(arr: np.ndarray) -> str:
    """Serialize numpy array to a simple string form stored in DB."""
    return "[" + " ".join(map(lambda x: repr(float(x)), arr.tolist())) + "]"


# -------------------------
# Build / (re)build annoy index
# -------------------------
def build_index_from_db(db_path: str = DB_PATH) -> Tuple[AnnoyIndex, Dict[int, str], int]:
    conn = get_conn(db_path)
    try:
        df = pd.read_sql_query("SELECT id, embedding FROM profile_embedding", conn)
    finally:
        conn.close()

    if df.empty:
        raise RuntimeError("profile_embedding table is empty or missing rows.")

    df["embedding_arr"] = df["embedding"].apply(str_to_array)
    df = df[df["embedding_arr"].notnull()].reset_index(drop=True)
    if df.empty:
        raise RuntimeError("No valid embeddings found in profile_embedding table.")

    vector_dim = int(len(df.loc[0, "embedding_arr"]))
    annoy_index = AnnoyIndex(vector_dim, "euclidean")
    id_map: Dict[int, str] = {}

    for idx, row in df.iterrows():
        emb = row["embedding_arr"]
        if emb is None or len(emb) != vector_dim:
            continue
        annoy_index.add_item(idx, emb.tolist())
        id_map[idx] = str(row["id"])

    annoy_index.build(10)
    return annoy_index, id_map, vector_dim


def ensure_index_loaded():
    global ANNOY_INDEX, ID_MAP, VECTOR_DIM
    with INDEX_LOCK:
        if ANNOY_INDEX is None:
            annoy_index, id_map, vector_dim = build_index_from_db(DB_PATH)
            ANNOY_INDEX = annoy_index
            ID_MAP = id_map
            VECTOR_DIM = vector_dim


# -------------------------
# Authentication dependency
# -------------------------
def authenticate(request: Request):
    auth = request.headers.get("Authorization")
    if not auth:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing Authorization header",
            headers={"WWW-Authenticate": "Bearer"},
        )
    token = auth.split(" ")[-1].strip()
    if token != EXPECTED_TOKEN:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not Authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return True


# -------------------------
# Robust interactions reader
# -------------------------
def get_interactions_df_normalized() -> pd.DataFrame:
    """
    Read interactions table (if present) and normalize to DataFrame with columns:
      - viewer_id
      - profile_id

    This function is robust: it inspects the table's columns and attempts to
    infer which column corresponds to viewer and profile using heuristics.
    If it can't find sensible columns, returns empty dataframe with the two columns.
    """
    if not os.path.exists(DB_PATH):
        return pd.DataFrame(columns=["viewer_id", "profile_id"])

    conn = get_conn(DB_PATH)
    try:
        # Check if interactions table exists
        cur = conn.cursor()
        cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='interactions'")
        if cur.fetchone() is None:
            return pd.DataFrame(columns=["viewer_id", "profile_id"])

        # read a sample (or full table if small)
        df = pd.read_sql_query("SELECT * FROM interactions LIMIT 10000", conn)
    finally:
        conn.close()

    if df.empty:
        return pd.DataFrame(columns=["viewer_id", "profile_id"])

    cols = list(df.columns)

    # heuristics for viewer column
    viewer_candidates = [c for c in cols if re.search(r"viewer|user", c, flags=re.I)]
    profile_candidates = [c for c in cols if re.search(r"profile|target|candidate|profile_id", c, flags=re.I)]

    viewer_col = viewer_candidates[0] if viewer_candidates else None
    profile_col = profile_candidates[0] if profile_candidates else None

    # fallback: common exact names
    if viewer_col is None and "viewer_id" in cols:
        viewer_col = "viewer_id"
    if profile_col is None and "profile_id" in cols:
        profile_col = "profile_id"

    # fallback: if still unknown, try by position (first two non-timestamp columns)
    if viewer_col is None or profile_col is None:
        # choose first two columns that are not obviously timestamps/date
        non_time = [c for c in cols if not re.search(r"time|date|timestamp|datetime", c, flags=re.I)]
        if len(non_time) >= 2:
            if viewer_col is None:
                viewer_col = non_time[0]
            if profile_col is None:
                profile_col = non_time[1]

    # If still missing, give up gracefully
    if viewer_col is None or profile_col is None:
        return pd.DataFrame(columns=["viewer_id", "profile_id"])

    # build normalized df
    normalized = pd.DataFrame()
    normalized["viewer_id"] = df[viewer_col].astype(str)
    normalized["profile_id"] = df[profile_col].astype(str)
    return normalized


def get_seen_profile_ids_for_viewer(viewer_id: str) -> Optional[List[str]]:
    """
    Return list of profile_id that this viewer has interacted with (or None).
    Normalizes viewer IDs by taking last segment after '-' (to match existing behavior).
    """
    df = get_interactions_df_normalized()
    if df.empty:
        return None
    df["viewer_norm"] = df["viewer_id"].apply(lambda w: str(w).split("-")[-1])
    matched = df.loc[df["viewer_norm"] == str(viewer_id), "profile_id"].tolist()
    return matched if matched else None


# -------------------------
# Helper: load embedding for a given profile id from DB
# -------------------------
def load_embedding_for_id(profile_id: str) -> Optional[np.ndarray]:
    conn = get_conn(DB_PATH)
    try:
        cur = conn.cursor()
        cur.execute("SELECT embedding FROM profile_embedding WHERE id = ?", (profile_id,))
        row = cur.fetchone()
        if row is None:
            return None
        return str_to_array(row["embedding"])
    finally:
        conn.close()


# -------------------------
# Endpoint: health
# -------------------------
@app.get("/")
def home():
    return {"status": "ok"}


# -------------------------
# Endpoint: match/{profile}
# -------------------------
@app.get("/match/{profile}")
def match(profile: str, n: int = 5):
    try:
        ensure_index_loaded()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to build index: {e}")

    emb = load_embedding_for_id(profile)
    if emb is None:
        raise HTTPException(status_code=404, detail=f"Embedding for profile id '{profile}' not found in DB.")

    seen_ids = get_seen_profile_ids_for_viewer(profile)

    with INDEX_LOCK:
        global ANNOY_INDEX, ID_MAP
        if ANNOY_INDEX is None:
            raise HTTPException(status_code=500, detail="Index not initialized")
        candidate_count = max(n + 10, n * 3)
        try:
            candidates = ANNOY_INDEX.get_nns_by_vector(emb.tolist(), candidate_count, include_distances=False)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Annoy query failed: {e}")

        result_ids: List[str] = []
        for idx in candidates:
            candidate_id = ID_MAP.get(idx)
            if candidate_id is None:
                continue
            if candidate_id == profile:
                continue
            if seen_ids is not None and candidate_id in seen_ids:
                continue
            result_ids.append(candidate_id)
            if len(result_ids) >= n:
                break

    return {"profile": result_ids}


# -------------------------
# Endpoint: update/{profile}
# -------------------------
@app.get("/update/{profile}")
def update(profile: str):
    seen = get_seen_profile_ids_for_viewer(profile)
    if not seen:
        return {"trained": "no", "reason": "no seen profiles found"}

    conn = get_conn(DB_PATH)
    try:
        placeholder = ",".join("?" for _ in seen)
        query = f"SELECT id, embedding FROM profile_embedding WHERE id IN ({placeholder})"
        df = pd.read_sql_query(query, conn, params=seen)
    finally:
        conn.close()

    if df.empty:
        return {"trained": "no", "reason": "no embeddings for seen profiles"}

    df["embedding_arr"] = df["embedding"].apply(str_to_array)
    valid_embs = df["embedding_arr"].dropna().tolist()
    if len(valid_embs) == 0:
        return {"trained": "no", "reason": "no valid embeddings for seen profiles"}

    stacked = np.vstack(valid_embs)
    avg = np.mean(stacked, axis=0)
    norm_val = np.linalg.norm(avg)
    if norm_val == 0:
        new_emb = avg
    else:
        new_emb = avg / norm_val

    serialized = array_to_str(new_emb)
    conn = get_conn(DB_PATH)
    try:
        cur = conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS profile_embedding (
                id TEXT PRIMARY KEY,
                embedding TEXT
            )
            """
        )
        cur.execute("SELECT 1 FROM profile_embedding WHERE id = ?", (profile,))
        exists = cur.fetchone() is not None
        if exists:
            cur.execute("UPDATE profile_embedding SET embedding = ? WHERE id = ?", (serialized, profile))
        else:
            cur.execute("INSERT INTO profile_embedding (id, embedding) VALUES (?, ?)", (profile, serialized))
        conn.commit()
    finally:
        conn.close()

    try:
        with INDEX_LOCK:
            annoy_index, id_map, vector_dim = build_index_from_db(DB_PATH)
            global ANNOY_INDEX, ID_MAP, VECTOR_DIM
            ANNOY_INDEX = annoy_index
            ID_MAP = id_map
            VECTOR_DIM = vector_dim
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to rebuild index after update: {e}")

    return {"trained": "ok"}


# -------------------------
# On startup ensure index loaded
# -------------------------
@app.on_event("startup")
def _startup():
    try:
        ensure_index_loaded()
        app.state.index_ready = True
    except Exception as e:
        app.state.index_ready = False
        app.state.index_error = str(e)