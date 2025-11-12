# app_sqlite.py — Streamlit dating prototype using SQLite for persistence
# Run with: streamlit run app_sqlite.py

import sqlite3
import json
import hashlib
from datetime import datetime
from urllib.parse import urljoin

import pandas as pd
import streamlit as st
import requests
import os

# ================================================================
# Config / constants
# ================================================================
GENDERS = ["Woman", "Man", "Non-binary"]
GRID_PAGE_SIZE_DEFAULT = 9

st.set_page_config(
    page_title="App Prototype (SQLite)",
    page_icon="💘",
    layout="wide",
)

# ================================================================
# Helpers for sqlite
# ================================================================
def get_conn(db_path: str):
    """Return sqlite3 connection (row factory as dict-like)."""
    conn = sqlite3.connect(db_path, timeout=10.0, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

def _as_json(value):
    return json.dumps(value, ensure_ascii=False)

def _parse_interests(val):
    if isinstance(val, list):
        return val
    try:
        x = json.loads(val)
        return x if isinstance(x, list) else []
    except Exception:
        pass
    s = str(val).strip()
    if s.startswith('[') and s.endswith(']'):
        inner = s[1:-1]
        parts = [p.strip().strip("'").strip('"') for p in inner.split(",") if p.strip()]
        return [p for p in parts if p]
    if "," in s:
        return [p.strip() for p in s.split(",") if p.strip()]
    return []

# ================================================================
# DB-backed persistence helpers
# ================================================================
def load_profiles_from_db(db_path: str) -> pd.DataFrame:
    """
    Load profiles table from sqlite into a DataFrame.
    Expected columns: id,name,age,gender,region,country,city,distance_km,interests,about,photo_url
    """
    if not os.path.exists(db_path):
        st.error(f"Database file not found at: {db_path}")
        return pd.DataFrame(columns=[
            "id","name","age","gender","region","country","city",
            "distance_km","interests","about","photo_url"
        ])
    try:
        conn = get_conn(db_path)
        q = "SELECT * FROM profiles"
        df = pd.read_sql_query(q, conn, params=None)
        conn.close()
    except Exception as e:
        st.error(f"Failed to read profiles table: {e}")
        return pd.DataFrame(columns=[
            "id","name","age","gender","region","country","city",
            "distance_km","interests","about","photo_url"
        ])

    # normalize and types
    if "id" in df.columns:
        df["id"] = df["id"].astype(str)
    if "age" in df.columns:
        df["age"] = pd.to_numeric(df["age"], errors="coerce").fillna(0).astype(int)
    if "distance_km" in df.columns:
        df["distance_km"] = pd.to_numeric(df["distance_km"], errors="coerce").fillna(0).astype(int)
    for col in ["region","country","city","about","photo_url"]:
        if col in df.columns:
            df[col] = df[col].fillna("")
    if "gender" in df.columns:
        df["gender"] = df["gender"].fillna("")
    # parse interests column into lists
    if "interests" in df.columns:
        df["interests"] = df["interests"].apply(_parse_interests)
    # ensure the expected order/columns
    cols = ["id","name","age","gender","region","country","city","distance_km","interests","about","photo_url"]
    for c in cols:
        if c not in df.columns:
            df[c] = "" if c not in ["age","distance_km","interests"] else 0
    return df[cols].copy()

def _load_viewers_df_from_db(db_path: str) -> pd.DataFrame:
    """Return viewers table as DataFrame (expected columns similar to VIEWER_COLS)."""
    default_cols = [
        "viewer_id","name","age","city",
        "seeking","age_min","age_max","top_interests",
        "w_age","w_distance","w_interests",
        "created_at","updated_at"
    ]
    if not os.path.exists(db_path):
        return pd.DataFrame(columns=default_cols)
    try:
        conn = get_conn(db_path)
        df = pd.read_sql_query("SELECT * FROM viewers", conn)
        conn.close()
    except Exception:
        return pd.DataFrame(columns=default_cols)

    # parse JSON-ish columns
    if "seeking" in df.columns:
        df["seeking"] = df["seeking"].apply(_parse_interests)
    if "top_interests" in df.columns:
        df["top_interests"] = df["top_interests"].apply(_parse_interests)
    return df

def upsert_viewer(settings: dict, viewer_id: str, db_path: str):
    """
    Insert or update a viewer in the viewers table.
    Creates the viewers table if missing.
    """
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    row = {
        "viewer_id": viewer_id,
        "name": settings.get("name", viewer_id),
        "age": int(settings.get("age", 0)),
        "city": settings.get("city", ""),
        "seeking": _as_json(settings.get("seeking", [])),
        "age_min": int(settings.get("age_min", 0)),
        "age_max": int(settings.get("age_max", 0)),
        "top_interests": _as_json(settings.get("top_interests", [])),
        "w_age": float(settings.get("weights", {}).get("age", 0.0)),
        "w_distance": float(settings.get("weights", {}).get("distance", 0.0)),
        "w_interests": float(settings.get("weights", {}).get("interests", 0.0)),
        "created_at": now,
        "updated_at": now,
    }
    os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
    conn = get_conn(db_path)
    cur = conn.cursor()
    # create table if not exists
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS viewers (
            viewer_id TEXT PRIMARY KEY,
            name TEXT,
            age INTEGER,
            city TEXT,
            seeking TEXT,
            age_min INTEGER,
            age_max INTEGER,
            top_interests TEXT,
            w_age REAL,
            w_distance REAL,
            w_interests REAL,
            created_at TEXT,
            updated_at TEXT
        )
        """
    )
    # attempt to insert or update
    cur.execute("SELECT * FROM viewers WHERE viewer_id = ?", (viewer_id,))
    existing = cur.fetchone()
    if existing:
        # keep created_at if present
        created_at = existing["created_at"] if "created_at" in existing.keys() and existing["created_at"] else row["created_at"]
        cur.execute(
            """
            UPDATE viewers SET
                name = ?, age = ?, city = ?, seeking = ?, age_min = ?, age_max = ?,
                top_interests = ?, w_age = ?, w_distance = ?, w_interests = ?, updated_at = ?
            WHERE viewer_id = ?
            """,
            (
                row["name"], row["age"], row["city"], row["seeking"], row["age_min"], row["age_max"],
                row["top_interests"], row["w_age"], row["w_distance"], row["w_interests"], row["updated_at"],
                viewer_id
            )
        )
    else:
        cur.execute(
            """
            INSERT INTO viewers (viewer_id,name,age,city,seeking,age_min,age_max,top_interests,w_age,w_distance,w_interests,created_at,updated_at)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                row["viewer_id"], row["name"], row["age"], row["city"], row["seeking"], row["age_min"], row["age_max"],
                row["top_interests"], row["w_age"], row["w_distance"], row["w_interests"], row["created_at"], row["updated_at"]
            )
        )
    conn.commit()
    conn.close()

# INTERACTIONS helpers
INTERACTION_FIELDS = [
    "timestamp","viewer_id","viewer_name","profile_id",
    "profile_name","action","compatibility"
]

def read_interactions_df(db_path: str) -> pd.DataFrame:
    """
    Read interactions table into DataFrame. If missing, return empty df with expected columns.
    """
    if not os.path.exists(db_path):
        return pd.DataFrame(columns=INTERACTION_FIELDS)
    try:
        conn = get_conn(db_path)
        df = pd.read_sql_query("SELECT * FROM interactions", conn)
        conn.close()
        if df is None or df.empty:
            return pd.DataFrame(columns=INTERACTION_FIELDS)
        # ensure column set
        for c in INTERACTION_FIELDS:
            if c not in df.columns:
                df[c] = None
        return df[INTERACTION_FIELDS]
    except Exception:
        # fallback: no interactions table or malformed
        return pd.DataFrame(columns=INTERACTION_FIELDS)

def hydrate_interactions_for_viewer(viewer_id: str, db_path: str):
    df = read_interactions_df(db_path)
    if df.empty or "viewer_id" not in df.columns:
        return {"likes": [], "passes": [], "superlikes": []}
    sub = df[df["viewer_id"].astype(str) == str(viewer_id)]
    return {
        "likes": sub.loc[sub["action"] == "like", "profile_id"].astype(str).tolist(),
        "passes": sub.loc[sub["action"] == "pass", "profile_id"].astype(str).tolist(),
        "superlikes": sub.loc[sub["action"] == "superlike", "profile_id"].astype(str).tolist(),
    }

# ================================================================
# Call GET /match/{profile} helper (ENSURE GET)
# ================================================================
def call_match_endpoint_get(profile_id: str, endpoint_template: str) -> dict:
    if not endpoint_template:
        return {"ok": False, "error": "no endpoint template configured"}
    try:
        if "{profile}" in endpoint_template:
            url = endpoint_template.format(profile=profile_id)
        else:
            base = endpoint_template.rstrip("/") + "/"
            url = urljoin(base, str(profile_id).lstrip("/"))
        resp = requests.get(url, timeout=6.0)
        try:
            parsed = resp.json()
        except Exception:
            parsed = None
        return {"ok": True, "method": "GET", "url": url, "status_code": resp.status_code, "text": resp.text, "json": parsed}
    except Exception as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {str(exc)}"}

# ================================================================
# Call GET /update/{profile} helper (mirrors match helper)
# ================================================================
def call_update_endpoint_get(profile_id: str, endpoint_template: str) -> dict:
    profile_id = profile_id.split("-")[-1]
    if not endpoint_template:
        return {"ok": False, "error": "no update endpoint template configured"}
    try:
        if "{profile}" in endpoint_template:
            url = endpoint_template.format(profile=profile_id)
        else:
            base = endpoint_template.rstrip("/") + "/"
            url = urljoin(base, str(profile_id).lstrip("/"))
        resp = requests.get(url, timeout=10.0)
        try:
            parsed = resp.json()
        except Exception:
            parsed = None
        return {"ok": True, "method": "GET", "url": url, "status_code": resp.status_code, "text": resp.text, "json": parsed}
    except Exception as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {str(exc)}"}

# ================================================================
# Interaction logging — write to sqlite AND call GET /match/{profile}
# ================================================================
def log_interaction(viewer_key: str, viewer_name: str, profile_row: pd.Series, action: str, compatibility: float, db_path: str):
    """
    Persist to sqlite interactions table and then call GET /match/{profile_id} synchronously.
    Writes the debug result to st.session_state['last_match_call'].
    """
    os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
    row = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "viewer_id": str(viewer_key),
        "viewer_name": str(viewer_name),
        "profile_id": str(profile_row["id"]),
        "profile_name": str(profile_row.get("name","")),
        "action": str(action),
        "compatibility": float(compatibility) if compatibility is not None else None,
    }
    # ensure interactions table exists
    conn = get_conn(db_path)
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS interactions (
            timestamp TEXT,
            viewer_id TEXT,
            viewer_name TEXT,
            profile_id TEXT,
            profile_name TEXT,
            action TEXT,
            compatibility REAL
        )
        """
    )
    # insert
    try:
        cur.execute(
            "INSERT INTO interactions (timestamp,viewer_id,viewer_name,profile_id,profile_name,action,compatibility) VALUES (?,?,?,?,?,?,?)",
            (
                row["timestamp"], row["viewer_id"], row["viewer_name"], row["profile_id"],
                row["profile_name"], row["action"], row["compatibility"]
            )
        )
        conn.commit()
    except Exception as e:
        st.session_state["last_db_error"] = str(e)
    finally:
        conn.close()

    # Call the match endpoint using GET and path param construction
    match_template = st.session_state.get("interactions_webhook", "").strip()
    k = viewer_key.split("-")[-1]
    if match_template:
        result = call_match_endpoint_get(str(k), match_template)
        st.session_state["last_match_call"] = result
    else:
        st.session_state["last_match_call"] = {"ok": False, "error": "no match endpoint configured"}

# ================================================================
# Lightweight caching + helpers (unchanged)
# ================================================================
@st.cache_data(show_spinner=False)
def _file_mtime(path: str):
    try:
        return os.path.getmtime(path)
    except Exception:
        return None

@st.cache_data(show_spinner=True)
def load_profiles_cached(db_path: str, mtime) -> pd.DataFrame:
    return load_profiles_from_db(db_path)

def health_banner():
    dbp = st.session_state.db_path
    exists = "✅" if os.path.exists(dbp) else "⚠️"
    st.info(
        f"DB: {exists} {dbp}"
    )

# ================================================================
# Session state bootstrapping (sqlite defaults)
# ================================================================
def ensure_state():
    if "db_path" not in st.session_state:
        st.session_state.db_path = os.path.join(os.getcwd(), "data.db")

    if "profiles_df" not in st.session_state:
        mt = _file_mtime(st.session_state.db_path)
        st.session_state.profiles_df = load_profiles_cached(st.session_state.db_path, mt)

    if "users" not in st.session_state:
        st.session_state.users = {}

    if "active_user" not in st.session_state:
        st.session_state.users["Default"] = {
            "settings": {
                "name": "Default", "age": 28, "city": "Mumbai",
                "seeking": ["Woman","Man","Non-binary"],
                "age_min": 22, "age_max": 40,
                "top_interests": ["Music","Travel","Foodie"],
                "weights": {"age": 0.3, "distance": 0.2, "interests": 0.5},
            },
            "likes": [], "passes": [], "superlikes": [],
            "current_index": 0,
        }
        st.session_state.active_user = "Default"

    # default template uses {profile} placeholder (recommended)
    if "interactions_webhook" not in st.session_state:
        st.session_state.interactions_webhook = "http://127.0.0.1:8000/match/{profile}"

    # default update webhook
    if "update_webhook" not in st.session_state:
        st.session_state.update_webhook = "http://127.0.0.1:8000/update/{profile}"

    if "ranked_cache" not in st.session_state:
        st.session_state.ranked_cache = {}
    if "grid_page" not in st.session_state:
        st.session_state.grid_page = 1
    if "grid_page_size" not in st.session_state:
        st.session_state.grid_page_size = GRID_PAGE_SIZE_DEFAULT
    if "low_bandwidth" not in st.session_state:
        st.session_state.low_bandwidth = True

    if "last_match_call" not in st.session_state:
        st.session_state["last_match_call"] = {"ok": False, "error": "no calls yet"}
    if "last_update_call" not in st.session_state:
        st.session_state["last_update_call"] = {"ok": False, "error": "no calls yet"}

    rehydrate_current_viewer_merge()

def get_active():
    return st.session_state.users[st.session_state.active_user]

# Hydration helpers (unchanged)
def rehydrate_current_viewer_merge():
    vid = st.session_state.active_user
    u = st.session_state.users.get(vid)
    if not u:
        return
    disk = hydrate_interactions_for_viewer(vid, st.session_state.db_path)
    u["likes"] = sorted(set(u.get("likes", [])) | set(disk.get("likes", [])))
    u["passes"] = sorted(set(u.get("passes", [])) | set(disk.get("passes", [])))
    u["superlikes"] = sorted(set(u.get("superlikes", [])) | set(disk.get("superlikes", [])))

# Login-as helper
def switch_to_profile_as_viewer(profile_row: pd.Series):
    vname = f"{profile_row['name']}-{profile_row['id']}"
    st.session_state.users.setdefault(vname, {
        "settings": {
            "name": profile_row["name"],
            "age": int(profile_row["age"]),
            "city": profile_row.get("city", ""),
            "seeking": GENDERS[:],
            "age_min": max(18, int(profile_row["age"]) - 5),
            "age_max": min(80, int(profile_row["age"]) + 5),
            "top_interests": list(profile_row.get("interests", [])[:3]) if isinstance(profile_row.get("interests", []), list) else [],
            "weights": {"age": 0.3, "distance": 0.2, "interests": 0.5},
        },
        "likes": [], "passes": [], "superlikes": [],
        "current_index": 0,
    })
    upsert_viewer(st.session_state.users[vname]["settings"], viewer_id=vname, db_path=st.session_state.db_path)
    st.session_state.active_user = vname
    rehydrate_current_viewer_merge()
    st.session_state.grid_page = 1

# Scoring and ranking functions unchanged...
def _row_hash(d: dict) -> str:
    return hashlib.md5(json.dumps(d, sort_keys=True).encode("utf-8")).hexdigest()

def _age_score_vector(age_series: pd.Series, amin: int, amax: int) -> pd.Series:
    mid = (amin + amax) / 2.0
    spread = max((amax - amin) / 2.0, 1.0)
    inside = age_series.between(amin, amax)
    score = 1.0 - (age_series.astype(float) - mid).abs() / spread
    score = score.clip(lower=0.0, upper=1.0)
    score = score.where(inside, other=0.0)
    return score

def _distance_score_vector(d_km: pd.Series) -> pd.Series:
    return (1.0 - (d_km.astype(float) / 30.0)).clip(lower=0.0, upper=1.0)

def _interest_overlap_vector(interests_col: pd.Series, your_top: set) -> pd.Series:
    if not your_top:
        return pd.Series(0.0, index=interests_col.index)
    denom = float(len(your_top))
    vals = [
        (len(your_top.intersection(set(v if isinstance(v, list) else []))) / denom)
        for v in interests_col
    ]
    return pd.Series(vals, index=interests_col.index)

def _settings_fingerprint(settings: dict) -> str:
    payload = {
        "age_min": settings["age_min"],
        "age_max": settings["age_max"],
        "seeking": tuple(sorted(settings["seeking"])),
        "top_interests": tuple(sorted(settings.get("top_interests", []))),
        "weights": (
            round(float(settings["weights"]["age"]), 4),
            round(float(settings["weights"]["distance"]), 4),
            round(float(settings["weights"]["interests"]), 4),
        ),
    }
    return _row_hash(payload)

def _profiles_fingerprint(df: pd.DataFrame) -> str:
    if df.empty:
        return "empty"
    cols = ["id","age","gender","city","country","distance_km"]
    take = df[cols].astype(str)
    md5 = hashlib.md5()
    md5.update(str(len(df)).encode("utf-8"))
    sample = take.iloc[::max(len(take)//500, 1)].to_csv(index=False).encode("utf-8")
    md5.update(sample)
    return md5.hexdigest()

def get_ranked_profiles(raw_df: pd.DataFrame, settings: dict, sort_by: str, viewer_id: str) -> pd.DataFrame:
    if raw_df.empty:
        return raw_df.copy()
    key = (
        _profiles_fingerprint(raw_df),
        _settings_fingerprint(settings),
        sort_by,
        viewer_id,
    )
    cache = st.session_state.ranked_cache
    if key in cache:
        return cache[key]
    df = raw_df.copy()
    mask = df["gender"].isin(settings["seeking"]) & df["age"].between(settings["age_min"], settings["age_max"])
    filtered = df[mask].copy()
    if filtered.empty:
        filtered = df.copy()
    w = settings["weights"]
    age_s = _age_score_vector(filtered["age"], settings["age_min"], settings["age_max"])
    dist_s = _distance_score_vector(filtered["distance_km"])
    your_set = set(settings.get("top_interests", []) or [])
    int_s = _interest_overlap_vector(filtered["interests"], your_set)
    filtered["compatibility"] = (w["age"] * age_s + w["distance"] * dist_s + w["interests"] * int_s).round(3)
    if sort_by == "Best match":
        filtered = filtered.sort_values(by=["compatibility", "distance_km"], ascending=[False, True])
    elif sort_by == "Nearest":
        filtered = filtered.sort_values(by=["distance_km", "compatibility"], ascending=[True, False])
    else:
        filtered = filtered.sample(frac=1, random_state=42)
    out = filtered.reset_index(drop=True)
    cache[key] = out
    return out

# UI helpers
def profile_card(row, show_image=True):
    with st.container():
        c1, c2 = st.columns([1, 2], gap="large")
        with c1:
            if show_image and row.get("photo_url"):
                try:
                    st.image(row["photo_url"], width='content', caption=f"{row['name']}, {row['age']} • {row['gender']}")
                except Exception:
                    # fallback to no image if URL invalid
                    st.caption(f"{row['name']}, {row['age']} • {row['gender']}")
            else:
                st.caption(f"{row['name']}, {row['age']} • {row['gender']}")
            st.caption(f"📍 {row['city']} • ~{row['distance_km']} km away")
            st.progress(min(max(float(row.get("compatibility", 0.0)), 0.0), 1.0))
            st.caption(f"Compat: {row.get('compatibility', 0.0):.2f}")
        with c2:
            st.subheader(f"{row['name']}")
            st.write(row["about"])
            if isinstance(row["interests"], list):
                st.write("**Interests**:", ", ".join(row["interests"]))
            else:
                st.write("**Interests**:", "")

def action_bar(row, user_state):
    c1, c2, c3, c4 = st.columns([1,1,1,1])
    with c1:
        if st.button("👎 Pass", key=f"pass_{row['id']}"):
            if row["id"] not in user_state["passes"]:
                user_state["passes"].append(row["id"])
            log_interaction(st.session_state.active_user, user_state["settings"]["name"], row, "pass", row.get("compatibility", 0.0), st.session_state.db_path)
            user_state["current_index"] += 1
            rehydrate_current_viewer_merge()
            st.rerun()
    with c2:
        if st.button("⭐ Superlike", key=f"super_{row['id']}"):
            if row["id"] not in user_state["superlikes"]:
                user_state["superlikes"].append(row["id"])
            log_interaction(st.session_state.active_user, user_state["settings"]["name"], row, "superlike", row.get("compatibility", 0.0), st.session_state.db_path)
            user_state["current_index"] += 1
            rehydrate_current_viewer_merge()
            st.rerun()
    with c3:
        if st.button("❤️ Like", key=f"like_{row['id']}"):
            if row["id"] not in user_state["likes"]:
                user_state["likes"].append(row["id"])
            log_interaction(st.session_state.active_user, user_state["settings"]["name"], row, "like", row.get("compatibility", 0.0), st.session_state.db_path)
            user_state["current_index"] += 1
            rehydrate_current_viewer_merge()
            st.rerun()
    with c4:
        if st.button("👤 View as this person", key=f"viewas_single_{row['id']}"):
            switch_to_profile_as_viewer(row)

def export_buttons(df, viewer_name, user_state):
    like_ids = set(user_state["likes"])
    pass_ids = set(user_state["passes"])
    super_ids = set(user_state["superlikes"])
    def label_status(pid):
        if pid in super_ids:
            return "superlike"
        if pid in like_ids:
            return "like"
        if pid in pass_ids:
            return "pass"
        return "unseen"
    out = df.copy()
    out["status"] = out["id"].apply(label_status)
    out.insert(0, "viewer_user", viewer_name)
    csv_bytes = out.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Export CSV (this user)",
        csv_bytes,
        file_name=f"{viewer_name}_feedback_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv"
    )

# ================================================================
# App entry
# ================================================================
ensure_state()

st.title("Recommendation (SQLite)")
st.caption("Pick any profile — you instantly 'log in' as them. Interactions log to SQLite and call GET /match/{profile}.")
health_banner()

# Viewer selection UI
with st.container():
    st.subheader("Login as any profile")
    df_choices = st.session_state.profiles_df.reset_index(drop=True)
    if df_choices.empty:
        st.warning("No profiles loaded. Check your DB path in the sidebar and reload.")
    else:
        labels = [
            f"{r['name']} ({r['id']}) — {r['city']}, {r['country']}"
            for _, r in df_choices.iterrows()
        ]
        default_ix = st.session_state.get("pick_profile_ix", 0)
        default_ix = min(default_ix, len(labels) - 1)
        def _on_pick_profile_as_viewer():
            ix = st.session_state["pick_profile_ix"]
            pr = df_choices.iloc[ix]
            switch_to_profile_as_viewer(pr)
        st.selectbox(
            "Pick profile to log in as",
            options=list(range(len(labels))),
            index=default_ix,
            key="pick_profile_ix",
            format_func=lambda i: labels[i],
            on_change=_on_pick_profile_as_viewer,
        )

# Sidebar with template input & settings
with st.sidebar:
    st.header("Viewer Settings & DB")
    ustate = get_active()
    s = ustate["settings"]
    generator_interests = sorted({it for lst in st.session_state.profiles_df.get("interests", []) for it in (lst or [])}) if not st.session_state.profiles_df.empty else []
    dataset_cities = sorted(st.session_state.profiles_df["city"].dropna().unique().tolist()) if not st.session_state.profiles_df.empty else []
    default_city = s.get("city") if s.get("city") in dataset_cities else (dataset_cities[0] if dataset_cities else "Mumbai")
    s["name"] = st.text_input("Your name", s["name"])
    s["age"] = st.number_input("Your age", min_value=18, max_value=80, value=int(s["age"]), step=1)
    s["city"] = st.selectbox("Your city", dataset_cities or ["Mumbai"], index=(dataset_cities.index(default_city) if dataset_cities and default_city in dataset_cities else 0))
    s["seeking"] = st.multiselect("Show me", GENDERS, default=s["seeking"])
    c1, c2 = st.columns(2)
    with c1:
        s["age_min"] = st.number_input("Min age", 18, 80, int(s["age_min"]), step=1)
    with c2:
        s["age_max"] = st.number_input("Max age", 18, 80, int(s["age_max"]), step=1)
    st.markdown("**Top interests** (helps ranking)")
    default_interest_seed = [i for i in s.get("top_interests", []) if i in generator_interests][:5]
    fallback = generator_interests[:3] if generator_interests else []
    s["top_interests"] = st.multiselect("Pick up to 5", generator_interests, default=(default_interest_seed or fallback), max_selections=5)
    st.markdown("**Scoring weights**")
    age_w = st.slider("Age fit", 0.0, 1.0, float(s["weights"]["age"]), 0.05)
    dist_w = st.slider("Distance", 0.0, 1.0, float(s["weights"]["distance"]), 0.05)
    int_w = st.slider("Interests overlap", 0.0, 1.0, float(s["weights"]["interests"]), 0.05)
    total = age_w + dist_w + int_w or 1.0
    s["weights"] = {"age": age_w/total, "distance": dist_w/total, "interests": int_w/total}
    # persist viewer to DB
    upsert_viewer(s, viewer_id=st.session_state.active_user, db_path=st.session_state.db_path)

    st.divider()
    st.subheader("Database")
    st.caption("Set the path to your SQLite DB (default ./data.db)")
    st.text_input("SQLite DB path", key="db_path", value=st.session_state.get("db_path", "./data.db"))
    if st.button("🔄 Reload profiles from DB"):
        mt = _file_mtime(st.session_state.db_path)
        st.session_state.profiles_df = load_profiles_cached(st.session_state.db_path, mt)
        st.session_state.ranked_cache.clear()
        for uname in st.session_state.users:
            st.session_state.users[uname]["current_index"] = 0
        st.success(f"Loaded {len(st.session_state.profiles_df)} profiles from {st.session_state.db_path}")

    st.divider()
    st.subheader("Performance")
    st.session_state.low_bandwidth = st.checkbox("Low-bandwidth mode (hide images in Grid)", value=st.session_state.low_bandwidth)
    st.session_state.grid_page_size = st.number_input("Grid page size", 3, 30, st.session_state.grid_page_size, 3)

    st.divider()
    st.subheader("History")
    if st.button("↻ Reload history for active viewer"):
        rehydrate_current_viewer_merge()
        st.success("History reloaded from interactions table")

    st.divider()
    st.subheader("Match endpoint (path)")
    st.caption("Enter either a template with '{profile}' or a base path. Examples:\n"
               "`http://127.0.0.1:8000/match/{profile}` or `http://127.0.0.1:8000/match`")
    st.text_input("Interactions webhook URL (template or base)", key="interactions_webhook", value=st.session_state.get("interactions_webhook", "http://127.0.0.1:8000/match/{profile}"))
    st.caption("When you Like/Pass/Superlike the app will call GET /match/<profile_id> and show the response in Debug.")

    st.divider()
    st.subheader("Update endpoint (path)")
    st.caption("Enter either a template with '{profile}' or a base path. Example:\n"
               "`http://127.0.0.1:8000/update/{profile}`")
    st.text_input("Profile update URL (template or base)", key="update_webhook", value=st.session_state.get("update_webhook", "http://127.0.0.1:8000/update/{profile}"))
    st.caption("This will be called for the active viewer just before ranking recommendations so embeddings can be refreshed.")

    st.divider()
    st.subheader("Maintenance")
    if st.button("👀 Show saved viewers table"):
        dfv = _load_viewers_df_from_db(st.session_state.db_path)
        st.dataframe(dfv, width='content')

# Ranking & UI
sort_by = st.selectbox("Sort by", ["Best match", "Nearest", "Shuffle"], index=0)

# Call /update/{viewer} before ranking to refresh embedding (if configured)
viewer_id = st.session_state.active_user
update_template = st.session_state.get("update_webhook", "").strip()
if update_template:
    result = call_update_endpoint_get(str(viewer_id), update_template)
    st.session_state["last_update_call"] = result
else:
    st.session_state["last_update_call"] = {"ok": False, "error": "no update endpoint configured"}

df_ranked = get_ranked_profiles(st.session_state.profiles_df, get_active()["settings"], sort_by, st.session_state.active_user)

m1, m2, m3, m4 = st.columns(4)
m1.metric("Profiles available", len(df_ranked))
m2.metric("Likes", len(get_active()["likes"]))
m3.metric("Superlikes", len(get_active()["superlikes"]))
m4.metric("Passes", len(get_active()["passes"]))

tabs = st.tabs(["Browse", "Grid", "Likes & Passes", "Debug"])

with tabs[0]:
    st.subheader("Swipe-ish")
    idx = get_active()["current_index"]
    if idx >= len(df_ranked) or df_ranked.empty:
        st.success("You're all caught up! Adjust filters or reload profiles.")
    else:
        row = df_ranked.iloc[idx]
        profile_card(row, show_image=True)
        action_bar(row, get_active())

with tabs[1]:
    st.subheader("All Profiles (paginated)")
    if df_ranked.empty:
        st.info("No profiles to show. Reload your profiles DB or relax filters.")
    else:
        total = len(df_ranked)
        per_page = int(st.session_state.grid_page_size)
        total_pages = max((total + per_page - 1) // per_page, 1)
        left, mid, right = st.columns([1,2,1])
        with left:
            if st.button("⬅️ Prev", disabled=(st.session_state.grid_page <= 1)):
                st.session_state.grid_page = max(1, st.session_state.grid_page - 1)
                st.rerun()
        with mid:
            st.markdown(f"Page **{st.session_state.grid_page} / {total_pages}**  •  Showing **{per_page}** per page")
        with right:
            if st.button("Next ➡️", disabled=(st.session_state.grid_page >= total_pages)):
                st.session_state.grid_page = min(total_pages, st.session_state.grid_page + 1)
                st.rerun()
        start = (st.session_state.grid_page - 1) * per_page
        end = min(start + per_page, total)
        page_df = df_ranked.iloc[start:end]
        n_cols = 3
        rows = [page_df.iloc[i:i+n_cols] for i in range(0, len(page_df), n_cols)]
        for chunk in rows:
            cols = st.columns(n_cols)
            for col, (_, r) in zip(cols, chunk.iterrows()):
                with col:
                    with st.container():
                        if st.session_state.low_bandwidth:
                            st.write(f"**{r['name']}**, {r['age']} • {r['gender']}")
                        else:
                            if r.get("photo_url"):
                                try:
                                    st.image(r["photo_url"], width='content')
                                except Exception:
                                    st.write(f"**{r['name']}**, {r['age']} • {r['gender']}")
                            else:
                                st.write(f"**{r['name']}**, {r['age']} • {r['gender']}")
                        st.caption(f"📍 {r['city']} • ~{r['distance_km']} km • Compat {r['compatibility']:.2f}")
                        if isinstance(r["interests"], list):
                            st.caption(", ".join(r["interests"]))
                        c1, c2, c3 = st.columns([1,1,1])
                        with c1:
                            if st.button("❤️", key=f"grid_like_{st.session_state.active_user}_{r['id']}_{start}"):
                                if r["id"] not in get_active()["likes"]:
                                    get_active()["likes"].append(r["id"])
                                log_interaction(st.session_state.active_user, get_active()["settings"]["name"], r, "like", r.get("compatibility", 0.0), st.session_state.db_path)
                                rehydrate_current_viewer_merge()
                        with c2:
                            if st.button("👎", key=f"grid_pass_{st.session_state.active_user}_{r['id']}_{start}"):
                                if r["id"] not in get_active()["passes"]:
                                    get_active()["passes"].append(r["id"])
                                log_interaction(st.session_state.active_user, get_active()["settings"]["name"], r, "pass", r.get("compatibility", 0.0), st.session_state.db_path)
                                rehydrate_current_viewer_merge()
                        with c3:
                            if st.button("👤 View as", key=f"grid_viewas_{r['id']}_{start}"):
                                switch_to_profile_as_viewer(r)

with tabs[2]:
    st.subheader("Your Decisions")
    ustate = get_active()
    base_df = st.session_state.profiles_df
    liked_ids = set(ustate["likes"] + ustate["superlikes"])
    passed_ids = set(ustate["passes"])
    liked_df = base_df[base_df["id"].isin(liked_ids)].copy()
    passed_df = base_df[base_df["id"].isin(passed_ids)].copy()
    if not liked_df.empty:
        liked_df = liked_df.merge(df_ranked[["id","compatibility"]], on="id", how="left")
    if not passed_df.empty:
        passed_df = passed_df.merge(df_ranked[["id","compatibility"]], on="id", how="left")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### ❤️ Likes & ⭐ Superlikes")
        if liked_df.empty:
            st.caption("No likes yet.")
        for _, r in liked_df.iterrows():
            with st.container():
                comp = r.get("compatibility")
                comp_txt = f" — Compat {comp:.2f}" if pd.notna(comp) else ""
                st.write(f"**{r['name']}**, {r['age']} • {r['gender']}{comp_txt}")
                st.caption(f"📍 {r['city']} • ~{r['distance_km']} km")
    with c2:
        st.markdown("### 👎 Passes")
        if passed_df.empty:
            st.caption("No passes yet.")
        for _, r in passed_df.iterrows():
            with st.container():
                comp = r.get("compatibility")
                comp_txt = f" — Compat {comp:.2f}" if pd.notna(comp) else ""
                st.write(f"**{r['name']}**, {r['age']} • {r['gender']}{comp_txt}")
                st.caption(f"📍 {r['city']} • ~{r['distance_km']} km")
    st.divider()
    export_buttons(base_df, st.session_state.active_user, ustate)

with tabs[3]:
    st.subheader("Debug / Developer Hooks")
    st.write("**Active viewer settings**")
    st.json(get_active()["settings"])
    st.write("**Current dataset (ranked for this viewer) — showing first 200 rows**")
    st.dataframe(df_ranked.head(200), width='content')
    st.markdown("**Recent interactions (active viewer)**")
    interactions_df = read_interactions_df(st.session_state.db_path)
    if interactions_df.empty:
        st.caption("No interactions table or it's empty.")
    else:
        st.dataframe(interactions_df[interactions_df["viewer_id"].astype(str) == str(st.session_state.active_user)].sort_values("timestamp", ascending=False).head(25), width='content')

    st.divider()
    st.markdown("### Last update endpoint call result")
    st.caption("Shows the last GET /update/{profile} call result (method/url/status/text/json or error).")
    st.json(st.session_state.get("last_update_call", {"ok": False, "error": "no calls yet"}))

    st.divider()
    st.markdown("### Last match endpoint call result")
    st.caption("Shows the last GET /match/{profile} call result (method/url/status/text/json or error).")
    st.json(st.session_state.get("last_match_call", {"ok": False, "error": "no calls yet"}))

    st.divider()
    st.markdown("**Maintenance**")
    if st.button("Repair interactions table (if needed)"):
        # Ensure interactions table exists with the expected columns
        conn = get_conn(st.session_state.db_path)
        cur = conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS interactions (
                timestamp TEXT,
                viewer_id TEXT,
                viewer_name TEXT,
                profile_id TEXT,
                profile_name TEXT,
                action TEXT,
                compatibility REAL
            )
            """
        )
        conn.commit()
        conn.close()
        st.success("Ensured interactions table exists with expected schema.")

    st.info(
        "DB → "
        f"{st.session_state.db_path}."
    )