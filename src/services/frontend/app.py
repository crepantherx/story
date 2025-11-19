# streamlit_app.py
import os
import json
import hashlib
from urllib.parse import urljoin
from datetime import datetime, timezone
import random

import pandas as pd
import requests
import streamlit as st

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

# Required environment runtime config
API_BASE = os.environ.get("API_BASE", "http://api:8080")
if not API_BASE:
    st.error("API_BASE environment variable is required and must point to your backend (e.g. http://127.0.0.1:8000).")
    st.stop()

# ================================================================
# Columns / shapes
# ================================================================
VIEWER_COLS = [
    "viewer_id", "name", "age", "city",
    "seeking", "age_min", "age_max", "top_interests",
    "w_age", "w_distance", "w_interests",
    "created_at", "updated_at"
]

PROFILES_COLS = [
    "id", "name", "age", "gender", "region", "country", "city",
    "distance_km", "interests", "about", "photo_url"
]

INTERACTION_FIELDS = [
    "timestamp", "viewer_id", "viewer_name", "profile_id",
    "profile_name", "action", "compatibility"
]

# ================================================================
# Small helpers
# ================================================================
def _row_hash(d: dict) -> str:
    return hashlib.md5(json.dumps(d, sort_keys=True).encode("utf-8")).hexdigest()

def safe_rerun():
    """
    Try to programmatically rerun the Streamlit script using whichever API
    exists on the installed Streamlit version.
    """
    try:
        return st.experimental_rerun()
    except Exception:
        try:
            return st.rerun()
        except Exception:
            # Can't programmatically rerun; instruct user to refresh
            raise RuntimeError(
                "Programmatic rerun not available in this Streamlit installation. "
                "Please refresh the page manually or upgrade Streamlit (`pip install -U streamlit`)."
            )

# ================================================================
# API callers (used for saving viewers & interactions)
# ================================================================
def upsert_viewer_via_api(settings: dict, viewer_id: str) -> dict:
    url = urljoin(API_BASE.rstrip("/") + "/", "upsert_viewer")
    payload = {"viewer_id": viewer_id}
    # Map simple fields
    for f in ["name", "age", "city", "seeking", "age_min", "age_max", "top_interests"]:
        val = settings.get(f)
        if val is not None:
            payload[f] = val

    # Map weights -> w_age, w_distance, w_interests
    w = settings.get("weights") or {}
    if "age" in w:
        payload["w_age"] = float(w["age"])
    if "distance" in w:
        payload["w_distance"] = float(w["distance"])
    if "interests" in w:
        payload["w_interests"] = float(w["interests"])

    try:
        r = requests.post(url, json=payload, timeout=10.0)
    except Exception as e:
        raise RuntimeError(f"Failed to call upsert endpoint: {e}")

    if r.status_code >= 400:
        try:
            err = r.json()
        except Exception:
            err = r.text
        raise RuntimeError(f"Upsert failed ({r.status_code}): {err}")

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
        raise RuntimeError(f"Add interaction failed ({r.status_code}): {err}")

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
        raise RuntimeError(f"Failed to call backend get_viewer: {e}")

    if r.status_code == 404:
        return None
    if r.status_code >= 400:
        try:
            err = r.json()
        except Exception:
            err = r.text
        raise RuntimeError(f"Backend get_viewer error ({r.status_code}): {err}")

    try:
        return r.json()
    except Exception as e:
        raise RuntimeError(f"Failed to parse get_viewer response: {e}")

# ----------------------
# Ensure every /match call sends only the raw id portion (strip name- prefix)
# ----------------------
def _strip_name_prefix_from_id(maybe_id: str) -> str:
    """
    If the app uses composite keys like "Name-<id>", return the trailing id
    portion after the first dash. If no dash present, return the original string.
    """
    if not isinstance(maybe_id, str):
        return str(maybe_id)
    if "-" in maybe_id:
        return maybe_id.split("-", 1)[1]
    return maybe_id

def call_match_endpoint_get(profile_id: str, endpoint_template: str) -> dict:
    """
    Call the configured match endpoint for a given profile/viewer id.
    Always send only the raw id (strip name prefix if present).

    Returns a dict with keys: ok (bool), status_code (int), url, text, json, and
    for debugging includes original_viewer_key and sent_id where applicable.
    """
    if not endpoint_template:
        return {"ok": False, "error": "no endpoint template configured"}

    original_id = str(profile_id)
    send_id = _strip_name_prefix_from_id(original_id)

    def _build_url(pid: str):
        if "{profile}" in endpoint_template:
            return endpoint_template.format(profile=pid)
        base = endpoint_template.rstrip("/") + "/"
        return urljoin(base, str(pid).lstrip("/"))

    try:
        url = _build_url(send_id)
        resp = requests.get(url, timeout=6.0)
        try:
            parsed = resp.json()
        except Exception:
            parsed = None
        ok = resp.status_code < 400
        result = {
            "ok": ok,
            "method": "GET",
            "url": url,
            "status_code": resp.status_code,
            "text": resp.text,
            "json": parsed,
            "original_viewer_key": original_id,
            "sent_id": send_id,
        }

        # If server returns an error, we keep this response but the caller will
        # treat ok=False and fall back to local ordering. We do not send the
        # composite name-id to the server at any point.
        return result
    except Exception as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {str(exc)}", "original_viewer_key": original_id, "sent_id": send_id}

# ================================================================
# Remote profiles loader (only remote; no CSV fallback)
# ================================================================
@st.cache_data(show_spinner=True)
def fetch_profiles_remote(api_base: str) -> pd.DataFrame:
    """
    Calls {api_base}/get_profiles and returns a DataFrame matching PROFILES_COLS.
    """
    url = urljoin(api_base.rstrip("/") + "/", "get_profiles")
    try:
        r = requests.get(url, timeout=8.0)
        r.raise_for_status()
    except Exception as e:
        raise RuntimeError(f"Failed to fetch profiles from backend: {e}")

    try:
        rows = r.json() or []
    except Exception as e:
        raise RuntimeError(f"Failed to parse profiles response: {e}")

    df = pd.DataFrame(rows)
    # ensure columns exist
    for c in PROFILES_COLS:
        if c not in df.columns:
            if c in ["age", "distance_km"]:
                df[c] = 0
            elif c == "interests":
                df[c] = [[] for _ in range(len(df))]
            else:
                df[c] = ""

    # Normalize interests field to list
    def _norm_interests(v):
        if isinstance(v, list):
            return v
        if isinstance(v, str):
            s = v.strip()
            if not s:
                return []
            if s.startswith("[") and s.endswith("]"):
                try:
                    return json.loads(s)
                except Exception:
                    pass
            return [p.strip() for p in s.split(",") if p.strip()]
        return []

    df["interests"] = df["interests"].apply(_norm_interests)

    # Numeric columns
    df["age"] = pd.to_numeric(df["age"], errors="coerce").fillna(0).astype(int)
    df["distance_km"] = pd.to_numeric(df["distance_km"], errors="coerce").fillna(0).astype(int)

    # text columns
    for col in ["region", "country", "city", "about", "photo_url", "gender", "name"]:
        df[col] = df[col].fillna("")

    df["id"] = df["id"].astype(str)
    return df[PROFILES_COLS].copy()

# ================================================================
# Compute helper
# ================================================================
def compute_all_interests_from_profiles(df: pd.DataFrame) -> list:
    s = set()
    if "interests" in df.columns:
        for lst in df["interests"]:
            if isinstance(lst, list):
                s.update([str(x) for x in lst])
    return sorted(s)

# ================================================================
# Scoring & ranking functions (unchanged)
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
    cols = ["id", "age", "gender", "city", "country", "distance_km"]
    take = df[cols].astype(str)
    md5 = hashlib.md5()
    md5.update(str(len(df)).encode("utf-8"))
    sample = take.iloc[::max(len(take)//500, 1)].to_csv(index=False).encode("utf-8")
    md5.update(sample)
    return md5.hexdigest()

def get_ranked_profiles(raw_df: pd.DataFrame, settings: dict, viewer_id: str) -> pd.DataFrame:
    """
    Server-driven ranking: attempt to fetch ordered matches from the configured
    interactions_webhook (/match/<viewer_id> or template). If the server returns
    an ordered list of profile ids or profiles, use that ordering. Otherwise
    fall back to a deterministic local ordering (Nearest) purely as a safety
    fallback.
    """
    if raw_df.empty:
        return raw_df.copy()

    cache = st.session_state.ranked_cache
    key = (
        _profiles_fingerprint(raw_df),
        _settings_fingerprint(settings),
        viewer_id,
        st.session_state.get("interactions_webhook", ""),
    )
    if key in cache:
        return cache[key]

    df = raw_df.copy()

    # First attempt: call server match endpoint to get authoritative ordering
    endpoint_template = st.session_state.get("interactions_webhook", "").strip()
    ordered_ids = None
    try:
        if endpoint_template:
            # viewer_id may be composite "name-id"; call_match_endpoint_get will
            # strip to the raw id before building the URL.
            result = call_match_endpoint_get(str(viewer_id), endpoint_template)
            # Treat HTTP errors as failures.
            if not result.get("ok") or result.get("status_code", 0) >= 400:
                # record last match call for debugging and fall through to fallback
                st.session_state["last_match_call"] = result
                ordered_ids = None
            else:
                st.session_state["last_match_call"] = result
                j = result.get("json")
                if isinstance(j, list):
                    ordered_ids = [str(x) for x in j]
                elif isinstance(j, dict):
                    if "matches" in j and isinstance(j["matches"], list):
                        ordered_ids = [str(x) for x in j["matches"]]
                    elif "profiles" in j and isinstance(j["profiles"], list):
                        try:
                            server_df = pd.DataFrame(j["profiles"])
                            for c in PROFILES_COLS:
                                if c not in server_df.columns:
                                    if c in ["age", "distance_km"]:
                                        server_df[c] = 0
                                    elif c == "interests":
                                        server_df[c] = [[] for _ in range(len(server_df))]
                                    else:
                                        server_df[c] = ""
                            server_df["interests"] = server_df["interests"].apply(lambda v: v if isinstance(v, list) else [])
                            server_df["age"] = pd.to_numeric(server_df["age"], errors="coerce").fillna(0).astype(int)
                            server_df["distance_km"] = pd.to_numeric(server_df["distance_km"], errors="coerce").fillna(0).astype(int)
                            server_df["id"] = server_df["id"].astype(str)
                            ordered = server_df[PROFILES_COLS].copy().reset_index(drop=True)
                            cache[key] = ordered
                            return ordered
                        except Exception:
                            ordered_ids = None
                elif isinstance(j, (int, str)):
                    ordered_ids = [str(j)]
    except Exception as exc:
        # network / unexpected error — record and move to fallback
        st.session_state["last_match_call"] = {"ok": False, "error": str(exc)}
        ordered_ids = None

    if ordered_ids:
        # preserve order from server and append any remaining profiles
        id_set = set(ordered_ids)
        ordered_list = [pid for pid in ordered_ids if pid in set(df["id"].astype(str))]
        remaining = df[~df["id"].astype(str).isin(id_set)].copy()
        remaining = remaining.sort_values(by=["distance_km"]).reset_index(drop=True)
        ordered_df = pd.concat([
            df[df["id"].astype(str).isin(ordered_list)].set_index(df[df["id"].astype(str).isin(ordered_list)]["id"].astype(str)),
            remaining.set_index(remaining["id"].astype(str))
        ], axis=0, sort=False)
        ordered_ids_present = [pid for pid in ordered_list if pid in ordered_df.index]
        final = ordered_df.loc[ordered_ids_present + [i for i in ordered_df.index if i not in ordered_ids_present]].reset_index(drop=True)
        cache[key] = final[PROFILES_COLS].copy()
        return cache[key]

    # Fallback local ordering (Nearest) — used only if server doesn't provide ordering
    filtered = df.copy()
    filtered["compatibility"] = 0.0
    filtered = filtered.sort_values(by=["distance_km"]).reset_index(drop=True)
    cache[key] = filtered[PROFILES_COLS + ["compatibility"]].copy()
    return cache[key]

# ================================================================
# UI components
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
            # st.progress(row.get("compatibility", 0.0), text=f"Compat: {row.get('compatibility', 0.0):.2f}")
        with c2:
            st.subheader(f"{row['name']}")
            st.write(row.get("about", ""))
            if isinstance(row.get("interests"), list):
                st.write("**Interests**:", ", ".join(row["interests"]))
            else:
                st.write("**Interests**:")

def action_bar(row, user_state):
    # c1, c2, c3, c4 = st.columns([1, 1, 1, 1])
    # c1, c2, c3, c4 = st.columns([0.2, 0.2, 0.2, 0.2])
    spacer_left, col, spacer_right = st.columns([2, 1, 2])
    with col:
        c1, c2, c3, c4 = st.columns([1, 1, 1, 1])

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
    # with c4:
    #     if st.button("👤 View as this person", key=f"viewas_single_{row['id']}"):
    #         switch_to_profile_as_viewer(row)

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
# Interaction logging — calls API and match endpoint
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
        # profile_row["id"] is already a raw id; call_match_endpoint_get will also strip if needed.
        result = call_match_endpoint_get(str(profile_row["id"]), match_template)
        st.session_state["last_match_call"] = result
    else:
        st.session_state["last_match_call"] = {"ok": False, "error": "no match endpoint configured"}

# ================================================================
# Application state bootstrapping (remote-only profiles)
# ================================================================
def ensure_state():
    # Load remote profiles
    if "profiles_df" not in st.session_state:
        try:
            st.session_state.profiles_df = fetch_profiles_remote(API_BASE)
        except Exception as e:
            st.error(f"Failed to load profiles from backend: {e}")
            st.stop()

    # ensure there is at least one profile
    if st.session_state.profiles_df.empty:
        st.error("No profiles returned from backend. Ensure /get_profiles returns data.")
        st.stop()

    # Users container
    if "users" not in st.session_state:
        st.session_state.users = {}

    # Choose a random profile on first load to act as the logged-in viewer (no "Default")
    if "active_user" not in st.session_state:
        # pick a random profile row
        try:
            pr = st.session_state.profiles_df.sample(n=1, random_state=random.randint(1, 10**9)).iloc[0]
        except Exception:
            # fallback deterministic pick
            pr = st.session_state.profiles_df.iloc[0]
        vname = f"{pr['name']}-{pr['id']}"
        # create user state from profile
        st.session_state.users[vname] = {
            "settings": {
                "name": pr.get("name", vname),
                "age": int(pr.get("age", 28)) if pd.notna(pr.get("age", None)) else 28,
                "city": pr.get("city", ""),
                "seeking": GENDERS[:],
                "age_min": max(18, int(pr.get("age", 23)) - 5) if pd.notna(pr.get("age", None)) else 18,
                "age_max": min(80, int(pr.get("age", 23)) + 5) if pd.notna(pr.get("age", None)) else 40,
                "top_interests": list(pr.get("interests", [])[:3]) if isinstance(pr.get("interests", []), list) else [],
                "weights": {"age": 0.3, "distance": 0.2, "interests": 0.5},
            },
            "likes": [], "passes": [], "superlikes": [],
            "current_index": 0,
        }
        st.session_state.active_user = vname

    # interaction webhook default
    if "interactions_webhook" not in st.session_state:
        st.session_state.interactions_webhook = API_BASE.rstrip("/") + "/match/{profile}"

    # caches & UI state
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
    if "last_add_interaction" not in st.session_state:
        st.session_state["last_add_interaction"] = {"ok": False, "error": "no calls yet"}

    # Ensure active user exists server-side (upsert)
    try:
        upsert_viewer_via_api(st.session_state.users[st.session_state.active_user]["settings"], st.session_state.active_user)
    except Exception as e:
        st.error(f"Failed to upsert active viewer on startup: {e}")
        st.stop()

    # Hydrate interactions for active user from backend
    try:
        disk = hydrate_interactions_for_viewer_remote(st.session_state.active_user)
        u = st.session_state.users.get(st.session_state.active_user)
        u["likes"] = sorted(set(u.get("likes", [])) | set(disk.get("likes", [])))
        u["passes"] = sorted(set(u.get("passes", [])) | set(disk.get("passes", [])))
        u["superlikes"] = sorted(set(u.get("superlikes", [])) | set(disk.get("superlikes", [])))
    except Exception as e:
        st.error(f"Failed to load interaction history: {e}")
        st.stop()

ensure_state()

# ================================================================
# profile-as-viewer and rehydration helpers
# ================================================================
def switch_to_profile_as_viewer(profile_row: pd.Series):
    vname = f"{profile_row['name']}-{profile_row['id']}"
    st.session_state.users.setdefault(vname, {
        "settings": {
            "name": profile_row.get("name", vname),
            "age": int(profile_row.get("age", 28)) if pd.notna(profile_row.get("age", None)) else 28,
            "city": profile_row.get("city", ""),
            "seeking": GENDERS[:],
            "age_min": max(18, int(profile_row.get("age", 23)) - 5) if pd.notna(profile_row.get("age", None)) else 18,
            "age_max": min(80, int(profile_row.get("age", 23)) + 5) if pd.notna(profile_row.get("age", None)) else 40,
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
st.title("Realtime Interaction-Driven Recommendations")
# st.caption("Interactions and viewers persist via API endpoints only. Profiles are loaded from backend (Supabase).")

def health_banner():
    st.subheader("")
    ok_api = False
    ok_get_profiles = False
    try:
        r = requests.get(API_BASE.rstrip("/") + "/", timeout=3.0)
        ok_api = (r.status_code < 400)
    except Exception:
        ok_api = False

    try:
        r2 = requests.get(urljoin(API_BASE.rstrip("/") + "/", f"get_profiles"), timeout=3.0)
        ok_get_profiles = (r2.status_code < 400)
    except Exception:
        ok_get_profiles = False

    st.info(
        f"Backend API: {'✅' if ok_api else '⚠️'} {API_BASE}\n\n"
        f"Get profiles endpoint: {'✅' if ok_get_profiles else '⚠️'}"
    )



# Viewer selection UI
with st.container():
    # st.subheader("Login as any profile")
    df_choices = st.session_state.profiles_df.reset_index(drop=True)
    if df_choices.empty:
        st.warning("No profiles loaded. Check your backend get_profiles endpoint.")
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

# NOTE: Sidebar removed — client-side controls are intentionally disabled. The server
# is authoritative for ordering via the /match endpoint. This simplifies the UI and
# ensures ordering is not confused with local heuristics.

# Ranking & display — server-driven
df_ranked = get_ranked_profiles(st.session_state.profiles_df, st.session_state.users[st.session_state.active_user]["settings"], st.session_state.active_user)

m1, m2, m3, m4 = st.columns(4)
m1.metric("Profiles available", len(df_ranked))
m2.metric("Likes", len(st.session_state.users[st.session_state.active_user].get("likes", [])))
superlikes_count = len(st.session_state.users.get(st.session_state.active_user, {}).get("superlikes", []))
m3.metric("Superlikes", superlikes_count)
m4.metric("Passes", len(st.session_state.users[st.session_state.active_user].get("passes", [])))

tabs = st.tabs(["Browse", "Grid", "Likes & Passes", "Debug"])

with tabs[0]:
    # st.subheader("Swipe-ish — server-driven order")
    idx = st.session_state.users[st.session_state.active_user]["current_index"]
    if idx >= len(df_ranked) or df_ranked.empty:
        st.success("You're all caught up! Wait for the server to provide more matches or change active viewer.")
    else:
        row = df_ranked.iloc[idx]
        profile_card(row, show_image=True)
        action_bar(row, st.session_state.users[st.session_state.active_user])

with tabs[1]:
    st.subheader("All Profiles (paginated) — server order")
    if df_ranked.empty:
        st.info("No profiles to show. Reload your profiles from backend or wait for server matches.")
    else:
        total = len(df_ranked)
        per_page = int(st.session_state.grid_page_size)
        total_pages = max((total + per_page - 1) // per_page, 1)
        left, mid, right = st.columns([1, 2, 1])
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
                        # st.caption(f"📍 {r['city']} • ~{r['distance_km']} km • Compatibility {r.get('compatibility', 0.0):.2f}")
                        if isinstance(r["interests"], list):
                            st.caption(", ".join(r["interests"]))
                        c1, c2, c3 = st.columns([1, 1, 1])
                        with c1:
                            if st.button("❤️", key=f"grid_like_{st.session_state.active_user}_{r['id']}_{start}"):
                                if r["id"] not in st.session_state.users[st.session_state.active_user]["likes"]:
                                    st.session_state.users[st.session_state.active_user]["likes"].append(r["id"])
                                log_interaction(st.session_state.active_user, st.session_state.users[st.session_state.active_user]["settings"]["name"], r, "like", r.get("compatibility", 0.0))
                                rehydrate_current_viewer_merge()
                        with c2:
                            if st.button("👎", key=f"grid_pass_{st.session_state.active_user}_{r['id']}_{start}"):
                                if r["id"] not in st.session_state.users[st.session_state.active_user]["passes"]:
                                    st.session_state.users[st.session_state.active_user]["passes"].append(r["id"])
                                log_interaction(st.session_state.active_user, st.session_state.users[st.session_state.active_user]["settings"]["name"], r, "pass", r.get("compatibility", 0.0))
                                rehydrate_current_viewer_merge()
                        with c3:
                            if st.button("👤 View as", key=f"grid_viewas_{r['id']}_{start}"):
                                switch_to_profile_as_viewer(r)

with tabs[2]:
    st.subheader("Your Decisions")
    ustate = st.session_state.users[st.session_state.active_user]
    base_df = st.session_state.profiles_df
    liked_ids = set(ustate.get("likes", []) + ustate.get("superlikes", []))
    passed_ids = set(ustate.get("passes", []))
    liked_df = base_df[base_df["id"].isin(liked_ids)].copy()
    passed_df = base_df[base_df["id"].isin(passed_ids)].copy()
    if not liked_df.empty:
        liked_df = liked_df.merge(df_ranked[["id", "compatibility"]], on="id", how="left")
    if not passed_df.empty:
        passed_df = passed_df.merge(df_ranked[["id", "compatibility"]], on="id", how="left")
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
    st.json(st.session_state.users[st.session_state.active_user]["settings"])

    st.markdown("**Current dataset (ranked for this viewer) — showing first 200 rows**")
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
    )

# health_banner()