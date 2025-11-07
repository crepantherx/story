# app.py — Streamlit Dating Recommender (loads pre-generated profiles.csv)
# Run with: streamlit run app.py

import os
import csv
import json
import streamlit as st
import pandas as pd
from datetime import datetime

# ================================================================
# --- CONFIG ---
# ================================================================
PROFILES_CSV = os.environ.get("PROFILES_CSV", "/Users/sudhirsingh/PyCharmProjects/story/src/services/frontend/data/profiles.csv")
INTERACTIONS_CSV = os.environ.get("INTERACTIONS_CSV", "./data/interactions.csv")
VIEWERS_CSV = os.environ.get("VIEWERS_CSV", "/Users/sudhirsingh/PyCharmProjects/story/src/services/frontend/data/viewers.csv")
LIST_COLUMNS = ["languages", "interests"]

GENDERS = ["Woman", "Man", "Non-binary"]

# ================================================================
# --- HELPERS ---
# ================================================================
def _parse_list(val):
    if pd.isna(val):
        return []
    try:
        x = json.loads(val)
        return x if isinstance(x, list) else [str(x)]
    except Exception:
        return [s.strip() for s in str(val).split(",") if s.strip()]


def load_profiles_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    for col in LIST_COLUMNS:
        if col in df.columns:
            df[col] = df[col].apply(_parse_list)
    req = ["id","name","age","gender","region","country","city","distance_km",
           "interests","about","photo_url"]
    for c in req:
        if c not in df.columns:
            df[c] = None
    # Defensive conversions
    if "age" in df.columns:
        df["age"] = pd.to_numeric(df["age"], errors="coerce").fillna(0).astype(int)
    if "distance_km" in df.columns:
        df["distance_km"] = pd.to_numeric(df["distance_km"], errors="coerce").fillna(0).astype(int)
    return df


def randomuser_url(pid: str, gender: str) -> str:
    idx = int(str(pid), 16) % 100
    folder = "women" if (gender == "Woman" or (gender not in ["Woman","Man"] and idx % 2 == 0)) else "men"
    return f"https://randomuser.me/api/portraits/{folder}/{idx}.jpg"


def ensure_photo_urls(df: pd.DataFrame) -> pd.DataFrame:
    if "photo_url" not in df.columns:
        df["photo_url"] = None
    mask = df["photo_url"].isna() | (df["photo_url"] == "") | (df["photo_url"].astype(str).str.lower() == "nan")
    df.loc[mask, "photo_url"] = df.loc[mask].apply(lambda r: randomuser_url(str(r["id"]), str(r["gender"])), axis=1)
    return df


VIEWER_COLS = [
    "viewer_id","name","age","city",
    "seeking","age_min","age_max","top_interests",
    "w_age","w_distance","w_interests",
    "created_at","updated_at"
]


def _as_json(value):
    return json.dumps(value, ensure_ascii=False)


def _load_viewers_df(path):
    if os.path.exists(path):
        try:
            return pd.read_csv(path)
        except Exception:
            return pd.DataFrame(columns=VIEWER_COLS)
    return pd.DataFrame(columns=VIEWER_COLS)


def upsert_viewer(settings: dict, viewer_id: str, path: str):
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
    if "viewer_id" in df.columns and (df["viewer_id"] == viewer_id).any():
        row["created_at"] = df.loc[df["viewer_id"] == viewer_id, "created_at"].iloc[0]
        df.loc[df["viewer_id"] == viewer_id, row.keys()] = list(row.values())
    else:
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df.to_csv(path, index=False)


# ================================================================
# --- STATE & SESSION ---
# ================================================================
def ensure_state():
    if "profiles_df" not in st.session_state:
        if os.path.exists(PROFILES_CSV):
            df = load_profiles_csv(PROFILES_CSV)
            df = ensure_photo_urls(df)
            st.session_state.profiles_df = df
        else:
            st.error(f"Profiles CSV not found at '{PROFILES_CSV}'. Please generate it first.")
            st.session_state.profiles_df = pd.DataFrame([])

    if "users" not in st.session_state:
        st.session_state.users = {}

    if "active_user" not in st.session_state:
        st.session_state.users["Default"] = {
            "settings": {
                "name": "Default", "age": 28, "city": "Mumbai",
                "seeking": GENDERS[:],
                "age_min": 22, "age_max": 40,
                "top_interests": ["Music","Travel","Foodie"],
                "weights": {"age": 0.3, "distance": 0.2, "interests": 0.5},
            },
            "likes": [], "passes": [], "superlikes": [],
            "current_index": 0,
        }
        st.session_state.active_user = "Default"

    if "interactions_csv" not in st.session_state:
        st.session_state.interactions_csv = INTERACTIONS_CSV
    if "viewers_csv" not in st.session_state:
        st.session_state.viewers_csv = VIEWERS_CSV


def get_active():
    return st.session_state.users[st.session_state.active_user]


# ================================================================
# --- SWITCH VIEWER ---
# ================================================================
def switch_to_profile_as_viewer(profile_row: pd.Series):
    vname = f"{profile_row['name']}-{profile_row['id']}"
    st.session_state.users[vname] = {
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
    }
    upsert_viewer(st.session_state.users[vname]["settings"], viewer_id=vname, path=st.session_state.viewers_csv)
    st.session_state.active_user = vname
    st.rerun()


# ================================================================
# --- LOGGING ---
# ================================================================
INTERACTION_FIELDS = ["timestamp","viewer_id","viewer_name","profile_id","profile_name","action","compatibility"]

def log_interaction(viewer_key: str, viewer_name: str, profile_row: pd.Series, action: str, compatibility: float):
    path = st.session_state.get("interactions_csv", INTERACTIONS_CSV)
    exists = os.path.exists(path)
    row = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "viewer_id": viewer_key,
        "viewer_name": viewer_name,
        "profile_id": str(profile_row["id"]),
        "profile_name": str(profile_row.get("name","")),
        "action": action,
        "compatibility": float(compatibility) if compatibility is not None else None,
    }
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=INTERACTION_FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


# ================================================================
# --- RECOMMENDATION LOGIC ---
# ================================================================
def compute_compatibility(row, settings):
    w = settings["weights"]
    target_min = settings["age_min"]
    target_max = settings["age_max"]
    if row["age"] < target_min or row["age"] > target_max:
        age_score = 0.0
    else:
        mid = (target_min + target_max) / 2.0
        spread = max((target_max - target_min) / 2.0, 1.0)
        age_score = max(0.0, 1.0 - abs(row["age"] - mid) / spread)
    dist = row["distance_km"]
    distance_score = max(0.0, 1.0 - (dist / 30.0))
    your_interests = set(get_active()["settings"]["top_interests"])
    their_interests = set(row["interests"]) if isinstance(row["interests"], list) else set()
    overlap = (len(your_interests & their_interests) / len(your_interests)) if your_interests else 0.0
    score = w["age"] * age_score + w["distance"] * distance_score + w["interests"] * overlap
    return round(float(score), 3)


def filtered_sorted_profiles(df, settings, sort_by="Best match"):
    mask = df["gender"].isin(settings["seeking"]) & df["age"].between(settings["age_min"], settings["age_max"])
    filtered = df[mask].copy()
    filtered["compatibility"] = filtered.apply(lambda r: compute_compatibility(r, settings), axis=1)
    if sort_by == "Best match":
        filtered = filtered.sort_values(by=["compatibility","distance_km"], ascending=[False,True])
    elif sort_by == "Nearest":
        filtered = filtered.sort_values(by=["distance_km","compatibility"], ascending=[True,False])
    else:
        filtered = filtered.sample(frac=1, random_state=42)
    return filtered.reset_index(drop=True)


# ================================================================
# --- UI HELPERS ---
# ================================================================
def profile_card(row):
    with st.container(border=True):
        c1, c2 = st.columns([1,2], gap="large")
        with c1:
            st.image(row["photo_url"], width="stretch", caption=f"{row['name']}, {row['age']} • {row['gender']}")
            st.caption(f"📍 {row['city']} • ~{row['distance_km']} km away")
            st.progress(row.get("compatibility", 0.0), text=f"Compat: {row.get('compatibility',0.0):.2f}")
        with c2:
            st.subheader(row["name"])
            st.write(row["about"])
            if isinstance(row["interests"], list):
                st.write("**Interests:**", ", ".join(row["interests"]))


def action_bar(row, user_state):
    c1, c2, c3, c4 = st.columns([1,1,1,1])
    with c1:
        if st.button("👎 Pass", key=f"pass_{row['id']}"):
            user_state["passes"].append(row["id"])
            log_interaction(st.session_state.active_user, user_state["settings"]["name"], row, "pass", row.get("compatibility",0.0))
            user_state["current_index"] += 1
            st.rerun()
    with c2:
        if st.button("⭐ Superlike", key=f"super_{row['id']}"):
            user_state["superlikes"].append(row["id"])
            log_interaction(st.session_state.active_user, user_state["settings"]["name"], row, "superlike", row.get("compatibility",0.0))
            user_state["current_index"] += 1
            st.rerun()
    with c3:
        if st.button("❤️ Like", key=f"like_{row['id']}"):
            user_state["likes"].append(row["id"])
            log_interaction(st.session_state.active_user, user_state["settings"]["name"], row, "like", row.get("compatibility",0.0))
            user_state["current_index"] += 1
            st.rerun()
    with c4:
        if st.button("👤 View as this person", key=f"viewas_{row['id']}"):
            switch_to_profile_as_viewer(row)


# ================================================================
# --- APP LAYOUT ---
# ================================================================
st.set_page_config(page_title="Recommendation", page_icon="💘", layout="wide")

ensure_state()

st.title("Recommendation")
st.caption("Profiles loaded from your generated profiles.csv file")

# --- Viewer (Login as any profile) ---
with st.container(border=True):
    st.subheader("Login as any profile")
    df_choices = st.session_state.profiles_df.reset_index(drop=True)
    if df_choices.empty:
        st.warning("No profiles loaded.")
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

# --- Sidebar Settings ---
with st.sidebar:
    st.header("Viewer Settings")
    ustate = get_active()
    s = ustate["settings"]
    dataset_cities = sorted(st.session_state.profiles_df["city"].dropna().unique().tolist())
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
    all_interests = sorted({i for lst in st.session_state.profiles_df["interests"] for i in (lst if isinstance(lst, list) else [])})
    s["top_interests"] = st.multiselect("Pick up to 5", all_interests, default=s["top_interests"], max_selections=5)

    st.markdown("**Scoring weights**")
    age_w = st.slider("Age fit", 0.0, 1.0, float(s["weights"]["age"]), 0.05)
    dist_w = st.slider("Distance", 0.0, 1.0, float(s["weights"]["distance"]), 0.05)
    int_w = st.slider("Interests overlap", 0.0, 1.0, float(s["weights"]["interests"]), 0.05)
    total = age_w + dist_w + int_w or 1.0
    s["weights"] = {"age": age_w/total, "distance": dist_w/total, "interests": int_w/total}
    upsert_viewer(s, viewer_id=st.session_state.active_user, path=st.session_state.viewers_csv)

    st.divider()
    st.subheader("Reload dataset")
    if st.button("🔄 Reload profiles.csv"):
        if os.path.exists(PROFILES_CSV):
            st.session_state.profiles_df = ensure_photo_urls(load_profiles_csv(PROFILES_CSV))
            for uname in st.session_state.users:
                st.session_state.users[uname]["current_index"] = 0
            st.success("Reloaded profiles from CSV.")
        else:
            st.error(f"File not found: {PROFILES_CSV}")

# --- Main recommendation tabs ---
sort_by = st.selectbox("Sort by", ["Best match", "Nearest", "Shuffle"], index=0)
df_ranked = filtered_sorted_profiles(st.session_state.profiles_df, get_active()["settings"], sort_by=sort_by)

idx = get_active()["current_index"]
if idx >= len(df_ranked):
    st.success("You're all caught up! Adjust filters or reload profiles.")
else:
    row = df_ranked.iloc[idx]
    profile_card(row)
    action_bar(row, get_active())
