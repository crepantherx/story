from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional
from supabase import create_client, Client
import os
from datetime import datetime, timezone

app = FastAPI()

import sys
from supabase import create_client, Client

url: str = "https://tpquhacpoxoschgsarie.supabase.co"
key: str = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InRwcXVoYWNwb3hvc2NoZ3NhcmllIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NjMyMTk3NjksImV4cCI6MjA3ODc5NTc2OX0.T06IB1qnCr8eL1BCvuSypVkS7Cgeu5wdnE8QrSWmb-w"

if not url or not key:
    print("Please set SUPABASE_URL and SUPABASE_KEY environment variables.", file=sys.stderr)
    sys.exit(1)

supabase: Client = create_client(url, key)

@app.get("/")
def home():
    return {"status": "ok"}


@app.get("/match/{profile}")
def match(profile: str):
    def a(profile):
        from pinecone import Pinecone

        pc = Pinecone(api_key="pcsk_61UNS7_CWf4kKYfMMbpSf3HgMmtaqMtXZYwemNJRR7b7RUcKc5RioQgNbCdmWd5sCgRx73")

        index = pc.Index("profiles")

        response = index.fetch(ids=[profile])

        import pandas as pd

        # fetch interactions for viewer (safe normalization)
        resp = supabase.table("interactions") \
            .select("profile_id") \
            .filter("viewer_id", "ilike", f"%{profile}%") \
            .execute()

        if resp is None:
            interactions_list = []
        elif isinstance(resp, dict):
            interactions_list = resp.get("data") or []
        else:
            interactions_list = getattr(resp, "data", []) or []

        # extract profile_id values
        already_seen = [r.get("profile_id") for r in interactions_list if
                        r.get("profile_id")] if interactions_list else []

        result = index.query(
            vector=response.vectors[profile].values,
            top_k=1,
            include_metadata=True,
            filter={
                "profile_id": {"$nin": already_seen}
            }
        )

        top1 = result.matches[0]
        return top1.id

    return {"best_match": a(profile)}

class ViewerUpsert(BaseModel):
    viewer_id: str
    name: Optional[str] = None
    age: Optional[int] = None
    city: Optional[str] = None
    seeking: Optional[List[str]] = None
    age_min: Optional[int] = None
    age_max: Optional[int] = None
    top_interests: Optional[List[str]] = None
    w_age: Optional[float] = None
    w_distance: Optional[float] = None
    w_interests: Optional[float] = None
    # created_at optional: if provided it will be used for inserts (and overwritten on update if you include it)
    created_at: Optional[str] = None

class Interaction(BaseModel):
    timestamp: Optional[str] = None
    viewer_id: str
    viewer_name: str
    profile_id: str
    profile_name: str
    action: str
    compatibility: float

TABLE = "viewers"
TABLE_INTERACTIONS = "interactions"
CONFLICT_KEY = "viewer_id"

@app.post("/upsert_viewer")
def upsert_viewer(payload: ViewerUpsert):
    now = datetime.now(timezone.utc).isoformat()

    # 1. Check if viewer already exists
    try:
        resp = supabase.table(TABLE).select("created_at").eq("viewer_id", payload.viewer_id).maybe_single().execute()
    except APIError as e:
        raise HTTPException(status_code=500, detail=str(e))

    # normalize response -> existing_data will be either dict (row) or None
    existing_data = None
    if resp is None:
        existing_data = None
    elif isinstance(resp, dict):
        # supabase-py may return a dict with "data"
        existing_data = resp.get("data")
    else:
        # object-like response with attribute .data
        existing_data = getattr(resp, "data", None)

    # Build row dynamically (ignore None fields)
    row = {k: v for k, v in payload.dict().items() if v is not None}
    row["viewer_id"] = payload.viewer_id

    if existing_data:
        # UPDATE — preserve created_at
        row["created_at"] = existing_data.get("created_at")
        row["updated_at"] = now
    else:
        # INSERT — set both created_at and updated_at
        row["created_at"] = now
        row["updated_at"] = now

    # 2. Run the upsert
    try:
        resp = supabase.table(TABLE).upsert(row, on_conflict=CONFLICT_KEY).execute()
    except APIError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # For v2: response is successful if we reach here
    return {
        "status": "success",
        "viewer_id": payload.viewer_id,
        "data": resp.data
    }

@app.post("/add_interaction")
def add_interaction(payload: Interaction):
    ts = payload.timestamp or datetime.now(timezone.utc).isoformat()

    row = {
        "timestamp": ts,
        "viewer_id": payload.viewer_id,
        "viewer_name": payload.viewer_name,
        "profile_id": payload.profile_id,
        "profile_name": payload.profile_name,
        "action": payload.action,
        "compatibility": payload.compatibility,
    }

    try:
        resp = supabase.table(TABLE_INTERACTIONS).insert(row).execute()
    except APIError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return {
        "status": "success",
        "data": resp.data
    }

@app.post("/create_viewers_table")
def create_viewers_table():
    sql = """
    CREATE TABLE IF NOT EXISTS public.viewers (
      viewer_id TEXT PRIMARY KEY,
      name TEXT,
      age INTEGER,
      city TEXT,
      seeking TEXT[],
      age_min INTEGER,
      age_max INTEGER,
      top_interests TEXT[],
      w_age NUMERIC,
      w_distance NUMERIC,
      w_interests NUMERIC,
      created_at TIMESTAMPTZ,
      updated_at TIMESTAMPTZ
    );
    """

    try:
        resp = supabase.rpc("exec_sql", {"sql": sql}).execute()
    except APIError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return {"status": "success", "table": "viewers"}

@app.post("/create_interactions_table")
def create_interactions_table():
    sql = """
    CREATE TABLE IF NOT EXISTS public.interactions (
      timestamp TIMESTAMPTZ,
      viewer_id TEXT,
      viewer_name TEXT,
      profile_id TEXT,
      profile_name TEXT,
      action TEXT,
      compatibility NUMERIC
    );
    """

    try:
        resp = supabase.rpc("exec_sql", {"sql": sql}).execute()
    except APIError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return {"status": "success", "table": "interactions"}

@app.get("/get_interactions/{viewer_id}")
def get_interactions_viewer(viewer_id: str):
    """
    Returns a list of interactions for the viewer_id.
    Response: JSON list of objects {profile_id, action, timestamp, profile_name, viewer_name}
    """
    try:
        resp = supabase.table("interactions").select("profile_id,action,timestamp,profile_name,viewer_name").eq("viewer_id", viewer_id).order("timestamp", desc=True).execute()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    # supabase-py v2 returns resp.data on success
    return resp.data or []

@app.get("/get_viewer/{viewer_id}")
def get_viewer(viewer_id: str):
    """
    Returns a single viewer row or 404.
    """
    try:
        resp = supabase.table("viewers").select("*").eq("viewer_id", viewer_id).limit(1).maybe_single().execute()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    row = resp.data
    if not row:
        raise HTTPException(status_code=404, detail="viewer not found")
    return row

def _normalize_execute_response(resp):
    """
    Normalize the return value of supabase.execute() into a python list or None.
    Supabase client may return None, a dict {'data': [...]}, or an object with .data.
    """
    if resp is None:
        return None
    if isinstance(resp, dict):
        return resp.get("data")
    return getattr(resp, "data", None)

@app.get("/get_profiles")
def get_profiles():
    """
    Return list of profiles for Streamlit. Each profile should contain fields:
    id, name, age, gender, region, country, city, distance_km, interests, about, photo_url
    """
    try:
        resp = supabase.table("profiles").select(
            "id,name,age,gender,region,country,city,distance_km,interests,about,photo_url"
        ).execute()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to query profiles: {e}")

    data = _normalize_execute_response(resp) or []
    # ensure each row is a plain dict and convert any Postgres array types to python lists if needed
    out = []
    for r in data:
        if not isinstance(r, dict):
            continue
        # defensive normalization
        row = {
            "id": str(r.get("id", "")),
            "name": r.get("name", "") or "",
            "age": int(r.get("age")) if r.get("age") is not None else 0,
            "gender": r.get("gender") or "",
            "region": r.get("region") or "",
            "country": r.get("country") or "",
            "city": r.get("city") or "",
            # keep distance numeric
            "distance_km": int(r.get("distance_km")) if r.get("distance_km") is not None else 0,
            # interests might already be list, or comma string — normalize to list
            "interests": r.get("interests") if isinstance(r.get("interests"), list) else (
                r.get("interests").split(",") if isinstance(r.get("interests"), str) and r.get("interests").strip() else []
            ),
            "about": r.get("about") or "",
            "photo_url": r.get("photo_url") or "",
        }
        out.append(row)

    return out

@app.get("/sync_sql_and_vector_db")
def sync_sql_and_vector_db():
    import sys
    from supabase import create_client, Client

    # --- config / client ---
    url: str = "https://tpquhacpoxoschgsarie.supabase.co"
    key: str = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InRwcXVoYWNwb3hvc2NoZ3NhcmllIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NjMyMTk3NjksImV4cCI6MjA3ODc5NTc2OX0.T06IB1qnCr8eL1BCvuSypVkS7Cgeu5wdnE8QrSWmb-w"

    if not url or not key:
        print("Please set SUPABASE_URL and SUPABASE_KEY environment variables.", file=sys.stderr)
        sys.exit(1)

    supabase: Client = create_client(url, key)

    res = supabase.table("profiles").select("*").execute()

    import pandas as pd
    profiles = pd.DataFrame(res.data)

    cols_to_club = [c for c in profiles.columns if c != "id"]
    profiles['text'] = profiles[cols_to_club].to_dict(orient="records")

    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(profiles['text'], batch_size=128, convert_to_numpy=True, normalize_embeddings=True)
    profiles['embedding'] = list(embeddings)

    profiles = profiles[['id', 'embedding']]

    # safe_pinecone_upsert.py
    from pinecone import Pinecone
    import numpy as np
    import math
    import json
    import sys
    import time
    from typing import Iterable

    # --- CONFIG ---
    API_KEY = "pcsk_61UNS7_CWf4kKYfMMbpSf3HgMmtaqMtXZYwemNJRR7b7RUcKc5RioQgNbCdmWd5sCgRx73"  # replace with your key
    INDEX_NAME = "profiles"
    DIM = 384  # set appropriate dimension for your embeddings
    METRIC = "cosine"
    CLOUD = "aws"  # adjust if needed
    REGION = "us-east-1"  # adjust if needed

    MAX_BYTES = 4_194_304  # 4 MB Pinecone payload limit
    SAFETY_MARGIN = 0.85  # aim under the limit

    # your dataframe of profiles (must have .id and .embedding columns)
    b = profiles  # replace with the variable that holds your DataFrame

    # --- Pinecone client & ServerlessSpec import (try variants) ---
    pc = Pinecone(api_key=API_KEY)

    if INDEX_NAME in pc.list_indexes():
        pc.delete_index(INDEX_NAME)
        print(f"Deleted index: {INDEX_NAME}")
    else:
        print(f"Index '{INDEX_NAME}' does not exist. No action taken.")
    ServerlessSpec = None
    try:
        from pinecone import ServerlessSpec
        # print("Imported ServerlessSpec from pinecone")
    except Exception:
        try:
            from pinecone.core.client.models import ServerlessSpec
            # print("Imported ServerlessSpec from pinecone.core.client.models")
        except Exception:
            ServerlessSpec = None
            # print("ServerlessSpec not available; will try to create index without spec if needed")

    # helper to list index names in a few SDK variants
    def list_index_names(pc_client):
        try:
            res = pc_client.list_indexes()
        except Exception as e:
            # fallback: some clients may raise or return something else
            raise RuntimeError(f"list_indexes() failed: {e}") from e

        # if object has names() method (newer clients), use it
        if hasattr(res, "names") and callable(res.names):
            try:
                return list(res.names())
            except Exception:
                pass

        # otherwise, if it's already an iterable of strings
        if isinstance(res, (list, tuple, set)):
            return list(res)

        # last resort: try to coerce to list
        try:
            return list(res)
        except Exception:
            raise RuntimeError(f"Cannot parse list_indexes() result: {res}")

    # create index if missing (attempt a couple of signatures)
    def ensure_index(pc_client, name, dimension, metric, serverless_spec=None):
        names = list_index_names(pc_client)
        if name in names:
            print(f"Index '{name}' already exists.")
            return

        print(f"Index '{name}' not found. Creating...")

        # Try a few ways to create the index depending on SDK surface
        # Primary approach: pass dimension, metric, spec (if available)
        try:
            if serverless_spec is not None:
                pc_client.create_index(
                    name=name,
                    dimension=dimension,
                    metric=metric,
                    spec=serverless_spec
                )
            else:
                pc_client.create_index(
                    name=name,
                    dimension=dimension,
                    metric=metric
                )
            print("Index created successfully.")
            return
        except Exception as e:
            print("Primary create_index() attempt failed:", e)

        # Older/newer SDKs sometimes expect spec only or different args
        try:
            if serverless_spec is not None:
                pc_client.create_index(
                    name=name,
                    spec=serverless_spec
                )
                print("Index created successfully (spec-only).")
                return
        except Exception as e:
            print("Spec-only create_index() attempt failed:", e)

        raise RuntimeError("Could not create index with available create_index signatures. "
                           "Check Pinecone client version and permissions.")

    # --- prepare serverless spec if available ---
    spec_obj = None
    if ServerlessSpec is not None:
        try:
            spec_obj = ServerlessSpec(cloud=CLOUD, region=REGION)
        except Exception:
            # Some ServerlessSpec constructors may accept different kwargs; attempt common ones:
            try:
                spec_obj = ServerlessSpec(cloud=CLOUD, region=REGION)  # redundant but explicit
            except Exception:
                spec_obj = None

    # Ensure index exists
    ensure_index(pc, INDEX_NAME, DIM, METRIC, serverless_spec=spec_obj)

    # Obtain index handle
    index = pc.Index(INDEX_NAME)

    # --- helper: estimate JSON payload bytes per vector (conservative) ---
    def approx_vector_size_bytes(vector_id: str, embedding) -> int:
        emb_len = len(embedding)
        # float32 ~4 bytes; JSON textual representation costs more -> multiply conservatively
        bytes_for_embedding = emb_len * 4 * 3
        overhead = 200 + len(str(vector_id))
        return int(bytes_for_embedding + overhead)

    def make_batches(rows: Iterable, max_batch_bytes: int):
        batch = []
        batch_bytes = 0
        for vid, emb in rows:
            est = approx_vector_size_bytes(vid, emb)
            if est > max_batch_bytes and batch:
                yield batch
                batch = [(vid, emb)]
                batch_bytes = est
                continue

            if (batch_bytes + est) > max_batch_bytes:
                if batch:
                    yield batch
                batch = [(vid, emb)]
                batch_bytes = est
            else:
                batch.append((vid, emb))
                batch_bytes += est

        if batch:
            yield batch

    # prepare rows: convert embeddings to float32 lists to reduce size
    rows = []
    for _, row in profiles.iterrows():
        vid = str(row.id)
        emb = np.asarray(row.embedding, dtype=np.float32).tolist()
        rows.append((vid, emb))

    target_max = int(MAX_BYTES * SAFETY_MARGIN)

    # optional estimate to help choose initial batch sizes
    if rows:
        avg_est = sum(approx_vector_size_bytes(v, e) for v, e in rows) / len(rows)
        est_batch_size = max(1, int(target_max // avg_est))
        print(f"Estimated bytes/vector ~{int(avg_est)}; estimated safe batch size ~{est_batch_size}")

    # upload loop with automatic shrinking on payload errors
    uploaded = 0
    start = time.time()

    for i, batch in enumerate(make_batches(rows, target_max), start=1):
        vectors = [(vid, emb) for vid, emb in batch]
        attempt_batch = vectors
        while True:
            try:
                index.upsert(attempt_batch)
                uploaded += len(attempt_batch)
                print(f"Batch {i}: uploaded {len(attempt_batch)} vectors (total {uploaded})")
                break
            except Exception as e:
                msg = str(e).lower()
                # detect payload-too-large style errors
                if ("message length too large" in msg or "limit" in msg or "413" in msg or "400" in msg) and len(
                        attempt_batch) > 1:
                    new_size = max(1, len(attempt_batch) // 2)
                    attempt_batch = attempt_batch[:new_size]
                    print(f"Server rejected batch: shrinking and retrying with {new_size} vectors...")
                    time.sleep(0.5)
                    continue
                # if single vector is too large or another error, raise
                raise

    end = time.time()
    print(f"Done. Uploaded {uploaded} vectors in {end - start:.1f} sec.")
    return {"status": "synced"}