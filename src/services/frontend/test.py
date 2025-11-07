# app.py — Minimal Dating App Prototype (static profiles.csv, fast, persistent interactions, robust hydration)
# Run with: streamlit run app.py

import os
import csv
import json
import hashlib
from datetime import datetime

import pandas as pd
import streamlit as st

# ================================================================
# Config / constants
# ================================================================
GENDERS = ["Woman", "Man", "Non-binary"]
GRID_PAGE_SIZE_DEFAULT = 9  # number of cards shown in Grid per page

st.set_page_config(
    page_title="App Prototype",
    page_icon="💘",
    layout="wide",
)

# ================================================================
# Persistence helpers
# ================================================================
VIEWER_COLS = [
    "viewer_id","name","age","city",
    "seeking","age_min","age_max","top_interests",
    "w_age","w_distance","w_interests",
    "created_at","updated_at"
]

PROFILES_COLS = [
    "id","name","age","gender","region","country","city",
    "distance_km","interests","about","photo_url"
]

INTERACTION_FIELDS = [
    "timestamp","viewer_id","viewer_name","profile_id",
    "profile_name","action","compatibility"
]

def _as_json(value):
    return json.dumps(value, ensure_ascii=False)

def _parse_interests(val):
    """Parse interests from CSV cell into a python list[str]."""
    if isinstance(val, list):
        return val
    # Try JSON first
    try:
        x = json.loads(val)
        return x if isinstance(x, list) else []
    except Exception:
        pass
    # Fallback for python-list-like strings: "['A', 'B']"
    s = str(val).strip()
    if s.startswith('[') and s.endswith(']'):
        inner = s[1:-1]
        parts = [p.strip().strip("'").strip('"') for p in inner.split(",") if p.strip()]
        return [p for p in parts if p]
    # Comma-separated fallback
    if "," in s:
        return [p.strip() for p in s.split(",") if p.strip()]
    return []

def _load_viewers_df(path):
    if os.path.exists(path):
        try:
            return pd.read_csv(path)
        except Exception:
            return pd.DataFrame(columns=VIEWER_COLS)
    return pd.DataFrame(columns=VIEWER_COLS)

def _row_hash(d: dict) -> str:
    return hashlib.md5(json.dumps(d, sort_keys=True).encode("utf-8")).hexdigest()

def upsert_viewer(settings: dict, viewer_id: str, path: str):
    """Write viewer settings only if something actually changed (prevents disk churn)."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    df = _load_viewers_df(path)
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

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    if "viewer_id" in df.columns and (df["viewer_id"] == viewer_id).any():
        old = df.loc[df["viewer_id"] == viewer_id].iloc[0].to_dict()
        new = old.copy()
        new.update(row)
        new["created_at"] = old.get("created_at", row["created_at"])
        if _row_hash(new) == _row_hash(old):
            return  # nothing changed → skip write
        df.loc[df["viewer_id"] == viewer_id, new.keys()] = list(new.values())
    else:
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)

    df.to_csv(path, index=False)

def load_profiles(path: str) -> pd.DataFrame:
    """Load static profiles, parse interests to list[str], and enforce required columns."""
    if not os.path.exists(path):
        st.error(f"Profiles file not found at: {path}")
        return pd.DataFrame(columns=PROFILES_COLS)
    df = pd.read_csv(
        path,
        converters={"id": str, "interests": _parse_interests}
    )
    missing = [c for c in PROFILES_COLS if c not in df.columns]
    if missing:
        st.error(f"profiles.csv is missing columns: {missing}")
        return pd.DataFrame(columns=PROFILES_COLS)
    # Ensure types
    df["id"] = df["id"].astype(str)
    if "age" in df: df["age"] = pd.to_numeric(df["age"], errors="coerce").fillna(0).astype(int)
    if "distance_km" in df: df["distance_km"] = pd.to_numeric(df["distance_km"], errors="coerce").fillna(0).astype(int)
    # Small normalization to avoid NA surprises
    for col in ["region","country","city","about","photo_url"]:
        if col in df.columns:
            df[col] = df[col].fillna("")
    if "gender" in df.columns:
        df["gender"] = df["gender"].fillna("")
    return df[PROFILES_COLS].copy()

def compute_all_interests_from_profiles(df: pd.DataFrame) -> list:
    """Build the universe of interests present in profiles for sidebar selection."""
    s = set()
    if "interests" in df.columns:
        for lst in df["interests"]:
            if isinstance(lst, list):
                s.update([str(x) for x in lst])
    return sorted(s)

# ---------- robust interactions loader ----------
def read_interactions_df(path: str) -> pd.DataFrame:
    """
    Always return a DataFrame with INTERACTION_FIELDS columns (may be empty).
    Handles:
      - missing file
      - empty file
      - file without header (header=None) IF column count matches
      - files with unexpected schema → return empty safe DF
    """
    if not os.path.exists(path):
        return pd.DataFrame(columns=INTERACTION_FIELDS)

    # Try normal read
    try:
        df = pd.read_csv(path, dtype={"viewer_id": str, "profile_id": str})
        # If it's empty but has no columns, return empty with schema
        if df.empty and not set(INTERACTION_FIELDS).issubset(set(df.columns)):
            return pd.DataFrame(columns=INTERACTION_FIELDS)
        # If required columns are missing, try header=None path
        if not set(INTERACTION_FIELDS).issubset(set(df.columns)):
            # Try to read without header and fix columns if width matches
            df2 = pd.read_csv(path, header=None)
            if df2.shape[1] == len(INTERACTION_FIELDS):
                df2.columns = INTERACTION_FIELDS
                return df2
            else:
                # schema mismatch → safest is empty DF with correct columns
                return pd.DataFrame(columns=INTERACTION_FIELDS)
        return df
    except Exception:
        # Try header=None fallback
        try:
            df2 = pd.read_csv(path, header=None)
            if df2.shape[1] == len(INTERACTION_FIELDS):
                df2.columns = INTERACTION_FIELDS
                return df2
        except Exception:
            pass
        return pd.DataFrame(columns=INTERACTION_FIELDS)

def hydrate_interactions_for_viewer(viewer_id: str, path: str):
    """Return dict with likes/passes/superlikes lists for a viewer from interactions.csv."""
    df = read_interactions_df(path)
    if df.empty or "viewer_id" not in df.columns:
        return {"likes": [], "passes": [], "superlikes": []}
    sub = df[df["viewer_id"].astype(str) == str(viewer_id)]
    return {
        "likes": sub.loc[sub["action"] == "like", "profile_id"].astype(str).tolist(),
        "passes": sub.loc[sub["action"] == "pass", "profile_id"].astype(str).tolist(),
        "superlikes": sub.loc[sub["action"] == "superlike", "profile_id"].astype(str).tolist(),
    }

def log_interaction(viewer_key: str, viewer_name: str, profile_row: pd.Series, action: str, compatibility: float):
    path = st.session_state.get(
        "interactions_csv",
        "/Users/sudhirsingh/PyCharmProjects/story/src/services/frontend/data/interactions.csv"
    )
    exists = os.path.exists(path)
    row = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "viewer_id": str(viewer_key),
        "viewer_name": str(viewer_name),
        "profile_id": str(profile_row["id"]),
        "profile_name": str(profile_row.get("name","")),
        "action": str(action),
        "compatibility": float(compatibility) if compatibility is not None else None,
    }
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=INTERACTION_FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerow(row)

# ================================================================
# Lightweight caching (no DF hashing)
# ================================================================
@st.cache_data(show_spinner=False)
def _file_mtime(path: str):
    try:
        return os.path.getmtime(path)
    except Exception:
        return None

@st.cache_data(show_spinner=True)
def load_profiles_cached(path: str, mtime) -> pd.DataFrame:
    # mtime acts as a cache key so edits on disk invalidate the cache
    return load_profiles(path)

def health_banner():
    pcsv = st.session_state.profiles_csv
    icsv = st.session_state.interactions_csv
    vcsv = st.session_state.viewers_csv
    exists = lambda p: "✅" if os.path.exists(p) else "⚠️"
    st.info(
        f"Profiles: {exists(pcsv)} {pcsv}\n\n"
        f"Interactions: {exists(icsv)} {icsv}\n\n"
        f"Viewers: {exists(vcsv)} {vcsv}"
    )

# ================================================================
# Session state
# ================================================================

def ensure_state():
    """Bootstrap session state and auto-login to a specific profile (default: ssse1024)."""
    # -------- fixed paths --------
    if "interactions_csv" not in st.session_state:
        st.session_state.interactions_csv = "/Users/sudhirsingh/PyCharmProjects/story/src/services/frontend/data/interactions.csv"
    if "viewers_csv" not in st.session_state:
        st.session_state.viewers_csv = "/Users/sudhirsingh/PyCharmProjects/story/src/services/frontend/data/viewers.csv"
    if "profiles_csv" not in st.session_state:
        st.session_state.profiles_csv = "/Users/sudhirsingh/PyCharmProjects/story/src/services/frontend/data/profiles.csv"

    # -------- load static profiles (cached by mtime) --------
    if "profiles_df" not in st.session_state:
        mt = _file_mtime(st.session_state.profiles_csv)
        st.session_state.profiles_df = load_profiles_cached(st.session_state.profiles_csv, mt)

    # -------- per-viewer container --------
    if "users" not in st.session_state:
        st.session_state.users = {}

    # Provide a minimal default viewer (used only if target ID not found)
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

    # -------- attempt to auto-login to your specific profile --------
    try:
        TARGET_ID = os.environ.get("DEFAULT_VIEWER_ID", "ssse1024")
        df = st.session_state.profiles_df
        if not df.empty:
            target_row = df.loc[df["id"].astype(str) == str(TARGET_ID)]
            if not target_row.empty:
                r = target_row.iloc[0]
                vname = f"{r['name']}-{r['id']}"

                # create viewer state once
                if vname not in st.session_state.users:
                    st.session_state.users[vname] = {
                        "settings": {
                            "name": r["name"],
                            "age": int(r["age"]),
                            "city": r.get("city", ""),
                            "seeking": ["Woman","Man","Non-binary"],
                            "age_min": max(18, int(r["age"]) - 5),
                            "age_max": min(80, int(r["age"]) + 5),
                            "top_interests": list(r.get("interests", [])[:3]) if isinstance(r.get("interests", []), list) else [],
                            "weights": {"age": 0.3, "distance": 0.2, "interests": 0.5},
                        },
                        "likes": [], "passes": [], "superlikes": [],
                        "current_index": 0,
                    }
                    # persist viewer + hydrate any past interactions
                    upsert_viewer(st.session_state.users[vname]["settings"], viewer_id=vname, path=st.session_state.viewers_csv)
                    hist = hydrate_interactions_for_viewer(vname, st.session_state.interactions_csv)
                    st.session_state.users[vname]["likes"] = sorted(set(hist.get("likes", [])))
                    st.session_state.users[vname]["passes"] = sorted(set(hist.get("passes", [])))
                    st.session_state.users[vname]["superlikes"] = sorted(set(hist.get("superlikes", [])))

                # make this the active user
                st.session_state.active_user = vname
    except Exception:
        # If anything goes sideways (missing CSV, columns, etc.), we silently keep "Default"
        pass

    # -------- ranking cache & UI state --------
    if "ranked_cache" not in st.session_state:
        st.session_state.ranked_cache = {}
    if "grid_page" not in st.session_state:
        st.session_state.grid_page = 1
    if "grid_page_size" not in st.session_state:
        # uses your existing constant; change default if you want
        st.session_state.grid_page_size = GRID_PAGE_SIZE_DEFAULT
    if "low_bandwidth" not in st.session_state:
        st.session_state.low_bandwidth = True

    # -------- final: merge-on-every-run to keep counters in sync with disk --------
    rehydrate_current_viewer_merge()



def get_active():
    return st.session_state.users[st.session_state.active_user]

# ================================================================
# Hydration helpers
# ================================================================
def rehydrate_current_viewer_merge():
    """
    Read likes/passes/superlikes for the active viewer from interactions.csv every run
    and MERGE with in-memory lists so counters increment immediately and persist after restart.
    """
    vid = st.session_state.active_user
    u = st.session_state.users.get(vid)
    if not u:
        return
    disk = hydrate_interactions_for_viewer(vid, st.session_state.interactions_csv)
    # merge (union) with current session state
    u["likes"] = sorted(set(u.get("likes", [])) | set(disk.get("likes", [])))
    u["passes"] = sorted(set(u.get("passes", [])) | set(disk.get("passes", [])))
    u["superlikes"] = sorted(set(u.get("superlikes", [])) | set(disk.get("superlikes", [])))

# ================================================================
# Login-as helper
# ================================================================
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
    upsert_viewer(st.session_state.users[vname]["settings"], viewer_id=vname, path=st.session_state.viewers_csv)
    st.session_state.active_user = vname
    # merge with disk history immediately so counters reflect past data as soon as you switch
    rehydrate_current_viewer_merge()
    # reset grid paging
    st.session_state.grid_page = 1

# ================================================================
# Vectorized matching + fast caching
# ================================================================
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
    """Stable, tiny fingerprint of settings for caching."""
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
    """Small fingerprint of the base data without hashing list columns."""
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
    """Rank with a session-level cache keyed by small fingerprints (no DF hashing)."""
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

    # filter first
    mask = df["gender"].isin(settings["seeking"]) & df["age"].between(settings["age_min"], settings["age_max"])
    filtered = df[mask].copy()
    if filtered.empty:
        filtered = df.copy()

    # vectorized scoring
    w = settings["weights"]
    age_s = _age_score_vector(filtered["age"], settings["age_min"], settings["age_max"])
    dist_s = _distance_score_vector(filtered["distance_km"])
    your_set = set(settings.get("top_interests", []) or [])
    int_s = _interest_overlap_vector(filtered["interests"], your_set)

    filtered["compatibility"] = (w["age"] * age_s + w["distance"] * dist_s + w["interests"] * int_s).round(3)

    # sort
    if sort_by == "Best match":
        filtered = filtered.sort_values(by=["compatibility", "distance_km"], ascending=[False, True])
    elif sort_by == "Nearest":
        filtered = filtered.sort_values(by=["distance_km", "compatibility"], ascending=[True, False])
    else:  # Shuffle
        filtered = filtered.sample(frac=1, random_state=42)

    out = filtered.reset_index(drop=True)
    cache[key] = out
    return out

# ================================================================
# UI helpers
# ================================================================
def profile_card(row, show_image=True):
    with st.container(border=True):
        c1, c2 = st.columns([1, 2], gap="large")
        with c1:
            if show_image:
                st.image(row["photo_url"], width='stretch', caption=f"{row['name']}, {row['age']} • {row['gender']}")
            else:
                st.caption(f"{row['name']}, {row['age']} • {row['gender']}")
            st.caption(f"📍 {row['city']} • ~{row['distance_km']} km away")
            st.progress(row.get("compatibility", 0.0), text=f"Compat: {row.get('compatibility', 0.0):.2f}")
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
            log_interaction(st.session_state.active_user, user_state["settings"]["name"], row, "pass", row.get("compatibility", 0.0))
            user_state["current_index"] += 1
            rehydrate_current_viewer_merge()
            st.rerun()
    with c2:
        if st.button("⭐ Superlike", key=f"super_{row['id']}"):
            if row["id"] not in user_state["superlikes"]:
                user_state["superlikes"].append(row["id"])
            log_interaction(st.session_state.active_user, user_state["settings"]["name"], row, "superlike", row.get("compatibility", 0.0))
            user_state["current_index"] += 1
            rehydrate_current_viewer_merge()
            st.rerun()
    with c3:
        if st.button("❤️ Like", key=f"like_{row['id']}"):
            if row["id"] not in user_state["likes"]:
                user_state["likes"].append(row["id"])
            log_interaction(st.session_state.active_user, user_state["settings"]["name"], row, "like", row.get("compatibility", 0.0))
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
# App
# ================================================================
ensure_state()

st.title("Recommendation")
st.caption("Pick any profile — you instantly 'log in' as them. Interactions log to CSV.")
health_banner()

# --- Viewer (single control: login as ANY profile) ---
with st.container(border=True):
    st.subheader("Login as any profile")
    df_choices = st.session_state.profiles_df.reset_index(drop=True)
    if df_choices.empty:
        st.warning("No profiles loaded. Check your profiles.csv path in the sidebar and reload.")
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

# --- Sidebar: settings + paths + perf toggles ---
with st.sidebar:
    st.header("Viewer Settings")
    ustate = get_active()
    s = ustate["settings"]

    # interests universe derived from profiles file
    generator_interests = compute_all_interests_from_profiles(st.session_state.profiles_df)

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

    # persist viewer after any changes (write only if changed)
    upsert_viewer(s, viewer_id=st.session_state.active_user, path=st.session_state.viewers_csv)

    st.divider()
    st.subheader("Profiles")
    st.caption("Profiles are loaded from the static CSV below.")
    st.text_input("Profiles CSV path", key="profiles_csv", value=st.session_state.get("profiles_csv", ""))

    if st.button("🔄 Reload profiles from CSV"):
        mt = _file_mtime(st.session_state.profiles_csv)
        st.session_state.profiles_df = load_profiles_cached(st.session_state.profiles_csv, mt)
        # blow the ranking cache because base data changed
        st.session_state.ranked_cache.clear()
        # reset per-viewer indices since ranking may change
        for uname in st.session_state.users:
            st.session_state.users[uname]["current_index"] = 0
        st.success(f"Loaded {len(st.session_state.profiles_df)} profiles from {st.session_state.profiles_csv}")

    st.divider()
    st.subheader("Performance")
    st.session_state.low_bandwidth = st.checkbox("Low-bandwidth mode (hide images in Grid)", value=st.session_state.low_bandwidth)
    st.session_state.grid_page_size = st.number_input("Grid page size", 3, 30, st.session_state.grid_page_size, 3)

    st.divider()
    st.subheader("History")
    if st.button("↻ Reload history for active viewer"):
        rehydrate_current_viewer_merge()
        st.success("History reloaded from interactions.csv")

    st.divider()
    st.subheader("Logging")
    st.text_input("Interactions CSV path", key="interactions_csv", value=st.session_state.get("interactions_csv", "/Users/sudhirsingh/PyCharmProjects/story/src/services/frontend/data/interactions.csv"))
    if os.path.exists(st.session_state.interactions_csv):
        st.caption(f"Logging to: `{st.session_state.interactions_csv}` (exists)")
    else:
        st.caption(f"Will create: `{st.session_state.interactions_csv}`")

    st.divider()
    st.subheader("Personas (Viewers) persistence")
    st.text_input("Viewers CSV path", key="viewers_csv", value=st.session_state.get("viewers_csv", "/Users/sudhirsingh/PyCharmProjects/story/src/services/frontend/data/viewers.csv"))
    if os.path.exists(st.session_state.viewers_csv):
        st.caption(f"Persisting viewers to: `{st.session_state.viewers_csv}` (exists)")
    else:
        st.caption(f"Will create: `{st.session_state.viewers_csv}` on first save")
    if st.button("👀 Show saved viewers.csv"):
        st.dataframe(_load_viewers_df(st.session_state.viewers_csv), width='stretch')

# Ranking & stats (cached via fingerprints)
sort_by = st.selectbox("Sort by", ["Best match", "Nearest", "Shuffle"], index=0)
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
        st.info("No profiles to show. Reload your profiles CSV or relax filters.")
    else:
        # Pagination controls
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
                    with st.container(border=True):
                        if st.session_state.low_bandwidth:
                            st.write(f"**{r['name']}**, {r['age']} • {r['gender']}")
                        else:
                            st.image(r["photo_url"], width='stretch')
                            st.write(f"**{r['name']}**, {r['age']} • {r['gender']}")
                        st.caption(f"📍 {r['city']} • ~{r['distance_km']} km • Compat {r['compatibility']:.2f}")
                        if isinstance(r["interests"], list):
                            st.caption(", ".join(r["interests"]))
                        c1, c2, c3 = st.columns([1,1,1])
                        with c1:
                            if st.button("❤️", key=f"grid_like_{st.session_state.active_user}_{r['id']}_{start}"):
                                if r["id"] not in get_active()["likes"]:
                                    get_active()["likes"].append(r["id"])
                                log_interaction(st.session_state.active_user, get_active()["settings"]["name"], r, "like", r.get("compatibility", 0.0))
                                rehydrate_current_viewer_merge()
                        with c2:
                            if st.button("👎", key=f"grid_pass_{st.session_state.active_user}_{r['id']}_{start}"):
                                if r["id"] not in get_active()["passes"]:
                                    get_active()["passes"].append(r["id"])
                                log_interaction(st.session_state.active_user, get_active()["settings"]["name"], r, "pass", r.get("compatibility", 0.0))
                                rehydrate_current_viewer_merge()
                        with c3:
                            if st.button("👤 View as", key=f"grid_viewas_{r['id']}_{start}"):
                                switch_to_profile_as_viewer(r)

with tabs[2]:
    st.subheader("Your Decisions")
    ustate = get_active()
    base_df = st.session_state.profiles_df  # use the full dataset so history is visible regardless of filters
    liked_ids = set(ustate["likes"] + ustate["superlikes"])
    passed_ids = set(ustate["passes"])

    liked_df = base_df[base_df["id"].isin(liked_ids)].copy()
    passed_df = base_df[base_df["id"].isin(passed_ids)].copy()

    # Optionally compute compatibility for display (not required for showing)
    if not liked_df.empty:
        liked_df = liked_df.merge(
            df_ranked[["id","compatibility"]], on="id", how="left"
        )
    if not passed_df.empty:
        passed_df = passed_df.merge(
            df_ranked[["id","compatibility"]], on="id", how="left"
        )

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### ❤️ Likes & ⭐ Superlikes")
        if liked_df.empty:
            st.caption("No likes yet.")
        for _, r in liked_df.iterrows():
            with st.container(border=True):
                comp = r.get("compatibility")
                comp_txt = f" — Compat {comp:.2f}" if pd.notna(comp) else ""
                st.write(f"**{r['name']}**, {r['age']} • {r['gender']}{comp_txt}")
                st.caption(f"📍 {r['city']} • ~{r['distance_km']} km")
    with c2:
        st.markdown("### 👎 Passes")
        if passed_df.empty:
            st.caption("No passes yet.")
        for _, r in passed_df.iterrows():
            with st.container(border=True):
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
    st.dataframe(df_ranked.head(200), width='stretch')

    st.markdown("**Recent interactions (active viewer)**")
    interactions_df = read_interactions_df(st.session_state.interactions_csv)
    if interactions_df.empty:
        st.caption("No interactions file or it's empty.")
    else:
        st.dataframe(
            interactions_df[interactions_df["viewer_id"].astype(str) == str(st.session_state.active_user)]
            .sort_values("timestamp", ascending=False)
            .head(25),
            width='stretch'
        )

    # Quick repair if file has 6 cols without header, etc.
    st.divider()
    st.markdown("**Maintenance**")
    if st.button("Repair interactions.csv header (if needed)"):
        df0 = read_interactions_df(st.session_state.interactions_csv)
        # If it's empty we just ensure header exists by rewriting an empty file with header
        if df0.empty:
            with open(st.session_state.interactions_csv, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=INTERACTION_FIELDS)
                writer.writeheader()
            st.success("Rewrote empty interactions.csv with correct header.")
        else:
            # Ensure column order & presence; rewrite
            df_fixed = df0.copy()
            for col in INTERACTION_FIELDS:
                if col not in df_fixed.columns:
                    df_fixed[col] = None
            df_fixed = df_fixed[INTERACTION_FIELDS]
            df_fixed.to_csv(st.session_state.interactions_csv, index=False)
            st.success("interactions.csv header/columns repaired.")

    st.info(
        "Profiles → "
        f"{st.session_state.profiles_csv}; "
        "Interactions → "
        f"{st.session_state.interactions_csv}; "
        "Viewers → "
        f"{st.session_state.viewers_csv}."
    )
