# app.py — Streamlit app (endpoint-driven) — full fixed version
# Run with: API_BASE=http://127.0.0.1:8000 streamlit run app.py

import os
import json
import hashlib
import traceback
from datetime import datetime, timezone
from urllib.parse import urljoin
from typing import Any

import pandas as pd
import streamlit as st
import requests

# ================================================================
# Config / constants
# ================================================================
GENDERS = ["Woman", "Man", "Non-binary"]
GRID_PAGE_SIZE_DEFAULT = 9

st.set_page_config(
    page_title="App Prototype (Endpoint-driven)",
    page_icon="💘",
    layout="wide",
)

API_BASE = "http://0.0.0.0:8000"
if not API_BASE:
    st.error("API_BASE environment variable is required and must point to your backend (e.g. http://127.0.0.1:8000).")
    st.stop()

# ================================================================
# Fields and column constants
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

# ================================================================
# Small helpers
# ================================================================
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

def _ensure_list(x):
    if x is None:
        return None
    if isinstance(x, list):
        return x
    try:
        parsed = json.loads(x)
        if isinstance(parsed, list):
            return parsed
    except Exception:
        pass
    if isinstance(x, str) and "," in x:
        return [p.strip() for p in x.split(",") if p.strip()]
    return [x]

def _safe_int(v, default=None):
    if v is None:
        return default
    try:
        return int(v)
    except Exception:
        return default

def _safe_float(v, default=None):
    if v is None:
        return default
    try:
        return float(v)
    except Exception:
        return default

def _row_hash(d: dict) -> str:
    return hashlib.md5(json.dumps(d, sort_keys=True).encode("utf-8")).hexdigest()

# Streamlit versions differ; safe rerun tries experimental_rerun then falls back
def safe_rerun():
    try:
        st.experimental_rerun()
    except Exception:
        # force small session change so page refreshes
        st.session_state["_rerun_count"] = st.session_state.get("_rerun_count", 0) + 1
        return

def get_active():
    return st.session_state.users[st.session_state.active_user]

# ================================================================
# API helpers — upsert viewer, add interaction, fetch history
# ================================================================
def upsert_viewer_via_api(settings: dict, viewer_id: str) -> dict:
    """
    Calls POST {API_BASE}/upsert_viewer to upsert one viewer.
    Defensive about types to avoid 422 errors from backend.
    Maps settings['weights'] -> w_age, w_distance, w_interests.
    """
    url = urljoin(API_BASE.rstrip("/") + "/", "upsert_viewer")
    payload = {"viewer_id": viewer_id}

    # text fields
    for key in ("name", "city"):
        val = settings.get(key)
        if val is not None:
            payload[key] = val

    # ints
    age = _safe_int(settings.get("age"))
    if age is not None:
        payload["age"] = age
    age_min = _safe_int(settings.get("age_min"))
    if age_min is not None:
        payload["age_min"] = age_min
    age_max = _safe_int(settings.get("age_max"))
    if age_max is not None:
        payload["age_max"] = age_max

    # lists
    seeking = _ensure_list(settings.get("seeking"))
    if seeking is not None:
        payload["seeking"] = seeking
    top_interests = _ensure_list(settings.get("top_interests"))
    if top_interests is not None:
        payload["top_interests"] = top_interests

    # weights -> explicit numeric fields
    w = settings.get("weights") or {}
    if "age" in w:
        payload["w_age"] = _safe_float(w.get("age"), 0.0)
    if "distance" in w:
        payload["w_distance"] = _safe_float(w.get("distance"), 0.0)
    if "interests" in w:
        payload["w_interests"] = _safe_float(w.get("interests"), 0.0)

    # send request
    try:
        r = requests.post(url, json=payload, timeout=10.0)
    except Exception as e:
        raise RuntimeError(f"Failed to call upsert endpoint: {e}")

    # surface backend errors clearly
    if r.status_code >= 400:
        try:
            err = r.json()
        except Exception:
            err = r.text
        raise RuntimeError(f"Upsert failed ({r.status_code}): {err}\nPayload: {json.dumps(payload, default=str)}")

    try:
        return r.json()
    except Exception:
        return {"status": "success", "raw": r.text}

def add_interaction_via_api(row: dict) -> dict:
    url = urljoin(API_BASE.rstrip("/") + "/", "add_interaction")
    try:
        r = requests.post(url, json=row, timeout=6.0)
    except Exception as e:
        raise RuntimeError(f"Failed to call add_interaction endpoint: {e}")

    if r.status_code >= 400:
        try:
            err = r.json()
        except Exception:
            err = r.text
        raise RuntimeError(f"Add interaction failed ({r.status_code}): {err}\nPayload: {json.dumps(row, default=str)}")

    try:
        return r.json()
    except Exception:
        return {"status": "success", "raw": r.text}

def hydrate_interactions_for_viewer_remote(viewer_id: str):
    url = urljoin(API_BASE.rstrip("/") + "/", f"get_interactions/{viewer_id}")
    try:
        r = requests.get(url, timeout=6.0)
    except Exception as e:
        raise RuntimeError(f"Failed to call backend get_interactions: {e}")

    if r.status_code >= 400:
        try:
            err = r.json()
        except Exception:
            err = r.text
        raise RuntimeError(f"Backend get_interactions error ({r.status_code}): {err}")

    try:
        rows = r.json() or []
    except Exception as e:
        raise RuntimeError(f"Failed to parse get_interactions response: {e}")

    likes = [rr["profile_id"] for rr in rows if rr.get("action") == "like"]
    passes = [rr["profile_id"] for rr in rows if rr.get("action") == "pass"]
    superlikes = [rr["profile_id"] for rr in rows if rr.get("action") == "superlike"]
    return {"likes": likes, "passes": passes, "superlikes": superlikes}

def fetch_viewer_row(viewer_id: str) -> dict:
    url = urljoin(API_BASE.rstrip("/") + "/", f"get_viewer/{viewer_id}")
    try:
        r = requests.get(url, timeout=6.0)
    except Exception as e:
        raise RuntimeError(f"Failed to call get_viewer: {e}")

    if r.status_code == 404:
        return None
    if r.status_code >= 400:
        try:
            err = r.json()
        except Exception:
            err = r.text
        raise RuntimeError(f"get_viewer failed ({r.status_code}): {err}")

    try:
        return r.json()
    except Exception as e:
        raise RuntimeError(f"Failed to parse get_viewer response: {e}")

# ================================================================
# Match endpoint helper
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
# Interaction logging / UI flow
# ================================================================
def log_interaction(viewer_key: str, viewer_name: str, profile_row: pd.Series, action: str, compatibility: float):
    ts = datetime.now(timezone.utc).isoformat()
    row = {
        "timestamp": ts,
        "viewer_id": str(viewer_key),
        "viewer_name": str(viewer_name),
        "profile_id": str(profile_row["id"]),
        "profile_name": str(profile_row.get("name", "")),
        "action": str(action),
        "compatibility": float(compatibility) if compatibility is not None else None,
    }

    try:
        add_res = add_interaction_via_api(row)
        st.session_state["last_add_interaction"] = {"ok": True, "response": add_res}
    except Exception as e:
        st.session_state["last_add_interaction"] = {"ok": False, "error": str(e)}
        st.error(f"Failed to add interaction: {e}")
        return

    match_template = st.session_state.get("interactions_webhook", "").strip()
    if match_template:
        result = call_match_endpoint_get(str(profile_row["id"]), match_template)
        st.session_state["last_match_call"] = result
    else:
        st.session_state["last_match_call"] = {"ok": False, "error": "no match endpoint configured"}

# ================================================================
# Profiles CSV loader (local)
# ================================================================
@st.cache_data(show_spinner=False)
def _file_mtime(path: str):
    try:
        return os.path.getmtime(path)
    except Exception:
        return None

@st.cache_data(show_spinner=True)
def load_profiles_cached(path: str, mtime) -> pd.DataFrame:
    return load_profiles(path)

def load_profiles(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        st.error(f"Profiles file not found at: {path}")
        return pd.DataFrame(columns=PROFILES_COLS)
    df = pd.read_csv(path, converters={"id": str, "interests": _parse_interests})
    missing = [c for c in PROFILES_COLS if c not in df.columns]
    if missing:
        st.error(f"profiles.csv missing columns: {missing}")
        return pd.DataFrame(columns=PROFILES_COLS)
    df["id"] = df["id"].astype(str)
    if "age" in df:
        df["age"] = pd.to_numeric(df["age"], errors="coerce").fillna(0).astype(int)
    if "distance_km" in df:
        df["distance_km"] = pd.to_numeric(df["distance_km"], errors="coerce").fillna(0).astype(int)
    for col in ["region","country","city","about","photo_url"]:
        if col in df.columns:
            df[col] = df[col].fillna("")
    if "gender" in df.columns:
        df["gender"] = df["gender"].fillna("")
    return df[PROFILES_COLS].copy()

def compute_all_interests_from_profiles(df: pd.DataFrame) -> list:
    s = set()
    if "interests" in df.columns:
        for lst in df["interests"]:
            if isinstance(lst, list):
                s.update([str(x) for x in lst])
    return sorted(s)

# ================================================================
# Application state bootstrapping (wrapped with debug catcher)
# ================================================================
def ensure_state():
    # default CSV path and load local profiles
    if "profiles_csv" not in st.session_state:
        st.session_state.profiles_csv = "./profiles.csv"
    if "profiles_df" not in st.session_state:
        mt = _file_mtime(st.session_state.profiles_csv)
        st.session_state.profiles_df = load_profiles_cached(st.session_state.profiles_csv, mt)

    # user state
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

    if "interactions_webhook" not in st.session_state:
        st.session_state.interactions_webhook = API_BASE.rstrip("/") + "/match/{profile}"

    st.session_state.setdefault("ranked_cache", {})
    st.session_state.setdefault("grid_page", 1)
    st.session_state.setdefault("grid_page_size", GRID_PAGE_SIZE_DEFAULT)
    st.session_state.setdefault("low_bandwidth", True)
    st.session_state.setdefault("last_match_call", {"ok": False, "error": "no calls yet"})
    st.session_state.setdefault("last_add_interaction", {"ok": False, "error": "no calls yet"})

    # Ensure default active user exists server-side (upsert). If backend returns 422 or other,
    # raise exception so debug wrapper can show full error.
    upsert_viewer_via_api(st.session_state.users[st.session_state.active_user]["settings"], st.session_state.active_user)

    # Hydrate interactions from backend for active user
    disk = hydrate_interactions_for_viewer_remote(st.session_state.active_user)
    u = st.session_state.users.get(st.session_state.active_user)
    u["likes"] = sorted(set(u.get("likes", [])) | set(disk.get("likes", [])))
    u["passes"] = sorted(set(u.get("passes", [])) | set(disk.get("passes", [])))
    u["superlikes"] = sorted(set(u.get("superlikes", [])) | set(disk.get("superlikes", [])))

# ================= DEBUG WRAPPER for ensure_state =================
try:
    ensure_state()
except Exception as e:
    st.error("Startup error in ensure_state() — showing details below.")
    st.markdown("**Exception:**")
    st.code(f"{type(e).__name__}: {e}")
    st.markdown("**Traceback:**")
    st.code(traceback.format_exc())
    st.markdown("**Runtime diagnostics**")
    st.write({"API_BASE": API_BASE})
    st.write("Profiles CSV path (session_state):", st.session_state.get("profiles_csv"))
    p = st.session_state.get("profiles_csv")
    if p:
        try:
            st.write("profiles.csv exists:", os.path.exists(p))
            if os.path.exists(p):
                st.write("profiles.csv first 10 lines:")
                with open(p, "r", encoding="utf-8") as fh:
                    for i, ln in enumerate(fh):
                        st.write(ln.rstrip())
                        if i >= 9:
                            break
        except Exception as ex:
            st.write("Failed to read profiles.csv:", ex)
    st.stop()

# ================================================================
# Ranking & scoring functions (unchanged)
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
    vals = [(len(your_top.intersection(set(v if isinstance(v, list) else []))) / denom) for v in interests_col]
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

# ================================================================
# UI helpers & components
# ================================================================
def profile_card(row, show_image=True):
    with st.container():
        c1, c2 = st.columns([1, 2], gap="large")
        with c1:
            if show_image:
                try:
                    st.image(row["photo_url"], width='stretch', caption=f"{row['name']}, {row['age']} • {row['gender']}")
                except Exception:
                    st.caption(f"{row['name']}, {row['age']} • {row['gender']}")
            else:
                st.caption(f"{row['name']}, {row['age']} • {row['gender']}")
            st.caption(f"📍 {row['city']} • ~{row.get('distance_km', 0)} km away")
            st.progress(row.get("compatibility", 0.0), text=f"Compat: {row.get('compatibility', 0.0):.2f}")
        with c2:
            st.subheader(f"{row['name']}")
            st.write(row.get("about", ""))
            if isinstance(row.get("interests"), list):
                st.write("**Interests**:", ", ".join(row["interests"]))
            else:
                st.write("**Interests**:")

def action_bar(row, user_state):
    c1, c2, c3, c4 = st.columns([1,1,1,1])
    with c1:
        if st.button("👎 Pass", key=f"pass_{row['id']}"):
            if row["id"] not in user_state["passes"]:
                user_state["passes"].append(row["id"])
            log_interaction(st.session_state.active_user, user_state["settings"]["name"], row, "pass", row.get("compatibility", 0.0))
            user_state["current_index"] += 1
            rehydrate_current_viewer_merge()
            safe_rerun()
    with c2:
        if st.button("⭐ Superlike", key=f"super_{row['id']}"):
            if row["id"] not in user_state["superlikes"]:
                user_state["superlikes"].append(row["id"])
            log_interaction(st.session_state.active_user, user_state["settings"]["name"], row, "superlike", row.get("compatibility", 0.0))
            user_state["current_index"] += 1
            rehydrate_current_viewer_merge()
            safe_rerun()
    with c3:
        if st.button("❤️ Like", key=f"like_{row['id']}"):
            if row["id"] not in user_state["likes"]:
                user_state["likes"].append(row["id"])
            log_interaction(st.session_state.active_user, user_state["settings"]["name"], row, "like", row.get("compatibility", 0.0))
            user_state["current_index"] += 1
            rehydrate_current_viewer_merge()
            safe_rerun()
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
# Helpers to change active viewer & hydrate history via API
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

    try:
        upsert_viewer_via_api(st.session_state.users[vname]["settings"], viewer_id=vname)
    except Exception as e:
        st.error(f"Failed to upsert viewer: {e}")
        return

    st.session_state.active_user = vname

    try:
        disk = hydrate_interactions_for_viewer_remote(vname)
        u = st.session_state.users.get(vname)
        u["likes"] = sorted(set(u.get("likes", [])) | set(disk.get("likes", [])))
        u["passes"] = sorted(set(u.get("passes", [])) | set(disk.get("passes", [])))
        u["superlikes"] = sorted(set(u.get("superlikes", [])) | set(disk.get("superlikes", [])))
    except Exception as e:
        st.error(f"Failed to load interaction history for {vname}: {e}")

    st.session_state.grid_page = 1

def rehydrate_current_viewer_merge():
    vid = st.session_state.active_user
    u = st.session_state.users.get(vid)
    if not u:
        return
    try:
        disk = hydrate_interactions_for_viewer_remote(vid)
    except Exception:
        disk = {"likes": [], "passes": [], "superlikes": []}
    u["likes"] = sorted(set(u.get("likes", [])) | set(disk.get("likes", [])))
    u["passes"] = sorted(set(u.get("passes", [])) | set(disk.get("passes", [])))
    u["superlikes"] = sorted(set(u.get("superlikes", [])) | set(disk.get("superlikes", [])))

# ================================================================
# App UI / Main
# ================================================================
st.title("Recommendation (Endpoint-driven)")
st.caption("Interactions and viewers persist via API endpoints only. Profiles still load from local CSV.")

def health_banner():
    ok_api = False
    ok_get_interactions = False
    try:
        r = requests.get(API_BASE.rstrip("/") + "/", timeout=3.0)
        ok_api = (r.status_code < 400)
    except Exception:
        ok_api = False

    try:
        r2 = requests.get(urljoin(API_BASE.rstrip("/") + "/", f"get_interactions/{st.session_state.active_user}"), timeout=3.0)
        ok_get_interactions = (r2.status_code < 400)
    except Exception:
        ok_get_interactions = False

    st.info(
        f"Backend API: {'✅' if ok_api else '⚠️'} {API_BASE}\n\n"
        f"Get interactions endpoint: {'✅' if ok_get_interactions else '⚠️'}"
    )

health_banner()

# Viewer selection
with st.container():
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
        default_ix = min(default_ix, max(0, len(labels) - 1))
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

# Sidebar
with st.sidebar:
    st.header("Viewer Settings")
    ustate = get_active()
    s = ustate["settings"]
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

    if st.button("Save viewer to server"):
        try:
            res = upsert_viewer_via_api(s, st.session_state.active_user)
            st.success("Saved to server.")
        except Exception as e:
            st.error(f"Save failed: {e}")

    st.divider()
    st.subheader("Profiles")
    st.caption("Profiles are loaded from the static CSV below.")
    st.text_input("Profiles CSV path", key="profiles_csv", value=st.session_state.get("profiles_csv", "./profiles.csv"))
    if st.button("🔄 Reload profiles from CSV"):
        mt = _file_mtime(st.session_state.profiles_csv)
        st.session_state.profiles_df = load_profiles_cached(st.session_state.profiles_csv, mt)
        st.session_state.ranked_cache.clear()
        for uname in st.session_state.users:
            st.session_state.users[uname]["current_index"] = 0
        st.success(f"Loaded {len(st.session_state.profiles_df)} profiles from {st.session_state.profiles_csv}")

    st.divider()
    st.subheader("Performance")
    st.session_state.low_bandwidth = st.checkbox("Low-bandwidth mode (hide images in Grid)", value=st.session_state.low_bandwidth)
    st.session_state.grid_page_size = st.number_input("Grid page size", 3, 30, st.session_state.grid_page_size, 3)

    st.divider()
    st.subheader("Match endpoint (path)")
    st.caption("Enter either a template with '{profile}' or a base path. Example: http://127.0.0.1:8000/match/{profile}")
    st.text_input("Interactions webhook URL (template or base)", key="interactions_webhook", value=st.session_state.get("interactions_webhook"))
    st.caption("When you Like/Pass/Superlike the app will call GET /match/<profile_id> and show the response in Debug.")

    st.divider()
    st.subheader("Backend info")
    st.write(f"API base: `{API_BASE}`")

# Ranking & display
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
        total = len(df_ranked)
        per_page = int(st.session_state.grid_page_size)
        total_pages = max((total + per_page - 1) // per_page, 1)
        left, mid, right = st.columns([1,2,1])
        with left:
            if st.button("⬅️ Prev", disabled=(st.session_state.grid_page <= 1)):
                st.session_state.grid_page = max(1, st.session_state.grid_page - 1)
                safe_rerun()
        with mid:
            st.markdown(f"Page **{st.session_state.grid_page} / {total_pages}**  •  Showing **{per_page}** per page")
        with right:
            if st.button("Next ➡️", disabled=(st.session_state.grid_page >= total_pages)):
                st.session_state.grid_page = min(total_pages, st.session_state.grid_page + 1)
                safe_rerun()
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
                            try:
                                st.image(r["photo_url"], width='stretch')
                            except Exception:
                                pass
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
    st.dataframe(df_ranked.head(200), width='stretch')

    st.markdown("**Recent interactions (active viewer) — fetched from server**")
    try:
        r = requests.get(urljoin(API_BASE.rstrip("/") + "/", f"get_interactions/{st.session_state.active_user}"), timeout=4.0)
        if r.status_code >= 400:
            st.error(f"Failed to fetch interactions: {r.status_code} {r.text}")
        else:
            rows = r.json() or []
            if not rows:
                st.caption("No interactions yet for active viewer.")
            else:
                df_re = pd.DataFrame(rows)
                st.dataframe(df_re.head(50), width='stretch')
    except Exception as e:
        st.error(f"Failed to fetch interactions: {e}")

    st.divider()
    st.markdown("### Last match endpoint call result")
    st.json(st.session_state.get("last_match_call", {"ok": False, "error": "no calls yet"}))

    st.divider()
    st.markdown("**Maintenance**")
    if st.button("Reload interactions & viewer history from server for active viewer"):
        try:
            rehydrate_current_viewer_merge()
            st.success("Reloaded remote history.")
        except Exception as e:
            st.error(f"Reload failed: {e}")

    st.info(
        f"API base → {API_BASE}\n\n"
        f"Profiles CSV → {st.session_state.profiles_csv}"
    )
