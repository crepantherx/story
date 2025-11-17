from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional
from supabase import create_client, Client
import os
from datetime import datetime, timezone

app = FastAPI()

import sys
from supabase import create_client, Client

# --- config / client ---
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
CONFLICT_KEY = "viewer_id"  # must match the unique/primary key on the table

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

# --- add near other helpers in your FastAPI app (app.py) ---
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


# --- new endpoint ---
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