
# app.py — Minimal Dating App Prototype (Multi-User "View As" Edition)
# Run with: streamlit run app.py

import os
import io
import json
import ast
import streamlit as st
import pandas as pd
from datetime import datetime

# ----------------------------
# Page config
# ----------------------------
st.set_page_config(
    page_title="App Prototype",
    page_icon="💘",
    layout="wide",
)

# ================================================================
# Compatibility / lists, interest universe (consistent with your generator)
# ================================================================
GENDERS = ["Woman", "Man", "Non-binary"]
INTEREST_CLUSTERS = {
    "Active": ["Hiking", "Running", "Yoga", "Dancing", "Photography"],
    "Arts": ["Art", "Theatre", "Poetry", "Movies", "Music"],
    "Geek": ["Tech", "Gaming", "Startups", "Board Games"],
    "Social": ["Foodie", "Travel", "Standup Comedy", "Volunteering"],
    "Sports": ["Cricket", "Football", "Basketball"],
}
ALL_INTERESTS = sorted({i for v in INTEREST_CLUSTERS.values() for i in v})

# ----------------------------
# Utilities: load a pre-generated corpus (CSV/JSONL/Parquet) and parse list columns
# ----------------------------
def _parse_list_cell(val):
    if isinstance(val, list):
        return val
    if pd.isna(val):
        return []
    s = str(val)
    # Try JSON first
    try:
        x = json.loads(s)
        if isinstance(x, list):
            return x
    except Exception:
        pass
    # Fallback to literal eval
    try:
        x = ast.literal_eval(s)
        if isinstance(x, list):
            return x
    except Exception:
        pass
    # Last resort: single string -> list
    return [s]

def load_corpus_from_path(path: str) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    if ext in [".csv", ".txt"]:
        df = pd.read_csv(path)
    elif ext in [".jsonl", ".json"]:
        # Handle line-delimited JSON
        try:
            df = pd.read_json(path, lines=True)
        except ValueError:
            df = pd.read_json(path)
    elif ext in [".parquet", ".pq"]:
        # Requires pyarrow/fastparquet installed in the runtime
        df = pd.read_parquet(path)
    else:
        raise ValueError(f"Unsupported file extension: {ext}")
    # Parse list columns commonly present in your generator
    for col in ["languages", "interests"]:
        if col in df.columns:
            df[col] = df[col].apply(_parse_list_cell)
    return df

@st.cache_data(show_spinner=False)
def cache_load_corpus(path: str):
    df = load_corpus_from_path(path)
    return df

def set_corpus(df: pd.DataFrame, source_label: str):
    st.session_state.profiles_df = df
    st.session_state.corpus_meta = {
        "source": source_label,
        "rows": int(len(df)),
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "sample_ids": df["id"].head(5).tolist() if "id" in df.columns else []
    }

# ----------------------------
# Session state bootstrap (no auto-generation)
# ----------------------------
def ensure_state():
    if "users" not in st.session_state:
        st.session_state.users = {}
    if "active_user" not in st.session_state:
        # Bootstrap with three demo "viewer" personas (only settings; dataset is external)
        st.session_state.users["Aditi"] = {
            "settings": {"name":"Aditi","age":27,"city":"Mumbai","seeking":["Man"],
                         "age_min":24,"age_max":34,"top_interests":["Music","Travel","Foodie"],
                         "weights":{"age":0.3,"distance":0.2,"interests":0.5}},
            "likes": [], "passes": [], "superlikes": [], "current_index": 0
        }
        st.session_state.users["Rahul"] = {
            "settings": {"name":"Rahul","age":30,"city":"Bengaluru","seeking":["Woman"],
                         "age_min":23,"age_max":33,"top_interests":["Tech","Startups","Cricket"],
                         "weights":{"age":0.25,"distance":0.25,"interests":0.5}},
            "likes": [], "passes": [], "superlikes": [], "current_index": 0
        }
        st.session_state.users["Sam"] = {
            "settings": {"name":"Sam","age":29,"city":"Pune","seeking":["Woman","Non-binary","Man"],
                         "age_min":22,"age_max":40,"top_interests":["Art","Theatre","Poetry"],
                         "weights":{"age":0.2,"distance":0.4,"interests":0.4}},
            "likes": [], "passes": [], "superlikes": [], "current_index": 0
        }
        st.session_state.active_user = "Rahul"
    if "corpus_meta" not in st.session_state:
        st.session_state.corpus_meta = {}
    # profiles_df is only set after user loads a corpus

def get_active():
    return st.session_state.users[st.session_state.active_user]

# ----------------------------
# Matching logic (same as before; works on the loaded corpus)
# ----------------------------
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

    dist = row.get("distance_km", 30)
    distance_score = max(0.0, 1.0 - (float(dist) / 30.0))

    your_interests = set(settings.get("top_interests", []))
    their_interests = set(row.get("interests", []))
    overlap = (len(your_interests & their_interests) / len(your_interests)) if your_interests else 0.0

    score = w["age"] * age_score + w["distance"] * distance_score + w["interests"] * overlap
    return round(float(score), 3)

def filtered_sorted_profiles(df, settings, sort_by="Best match"):
    if df is None or df.empty:
        return df
    # Filter by gender and age if columns exist
    mask = pd.Series([True] * len(df))
    if "gender" in df.columns:
        mask &= df["gender"].isin(settings["seeking"])
    if "age" in df.columns:
        mask &= df["age"].between(settings["age_min"], settings["age_max"])
    filtered = df[mask].copy()
    # Compute compatibility
    filtered["compatibility"] = filtered.apply(lambda r: compute_compatibility(r, settings), axis=1)
    if sort_by == "Best match":
        filtered = filtered.sort_values(by=["compatibility", "distance_km"], ascending=[False, True], na_position="last")
    elif sort_by == "Nearest" and "distance_km" in filtered.columns:
        filtered = filtered.sort_values(by=["distance_km", "compatibility"], ascending=[True, False], na_position="last")
    else:
        filtered = filtered.sample(frac=1, random_state=42)
    return filtered.reset_index(drop=True)

def profile_card(row):
    with st.container(border=True):
        c1, c2 = st.columns([1, 2], gap="large")
        with c1:
            st.image(row.get("photo_url",""), use_container_width=True, caption=f"{row.get('name','')}, {row.get('age','—')} • {row.get('gender','')}")
            city = row.get("city", "")
            dist = row.get("distance_km", "—")
            st.caption(f"📍 {city} • ~{dist} km away")
            st.progress(row.get("compatibility", 0.0), text=f"Compat: {row.get('compatibility', 0.0):.2f}")
        with c2:
            st.subheader(f"{row.get('name','')}")
            st.write(row.get("about",""))
            ints = row.get("interests", [])
            if isinstance(ints, list):
                st.write("**Interests**:", ", ".join(ints))

def action_bar(row, user_state):
    c1, c2, c3, _ = st.columns([1,1,1,6])
    with c1:
        if st.button("👎 Pass", key=f"pass_{row['id']}"):
            user_state["passes"].append(row["id"])
            user_state["current_index"] += 1
            st.rerun()
    with c2:
        if st.button("⭐ Superlike", key=f"super_{row['id']}"):
            user_state["superlikes"].append(row["id"])
            user_state["current_index"] += 1
            st.rerun()
    with c3:
        if st.button("❤️ Like", key=f"like_{row['id']}"):
            user_state["likes"].append(row["id"])
            user_state["current_index"] += 1
            st.rerun()

def export_buttons(df, viewer_name, user_state):
    like_ids = set(user_state["likes"])
    pass_ids = set(user_state["passes"])
    super_ids = set(user_state["superlikes"])
    def label_status(pid):
        if pid in super_ids: return "superlike"
        if pid in like_ids: return "like"
        if pid in pass_ids: return "pass"
        return "unseen"
    out = df.copy()
    if "id" in out.columns:
        out["status"] = out["id"].apply(label_status)
    out.insert(0, "viewer_user", viewer_name)
    csv = out.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Export CSV (this user)",
        csv,
        file_name=f"{viewer_name}_feedback_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv"
    )

# ----------------------------
# App
# ----------------------------
ensure_state()

st.title("Recommendation")
st.caption("Bring your own corpus. The engine will rank and browse from your loaded profiles.")

# --- Corpus loader (sidebar) ---
with st.sidebar:
    st.header("Corpus")
    st.caption("Load your pre-generated profile corpus (CSV / JSONL / Parquet). No auto-generation.")

    # Option A: File uploader
    up = st.file_uploader("Upload corpus file", type=["csv","jsonl","json","parquet","pq"])
    if up is not None:
        # Write to a temp path under .streamlit cache; then load through pandas
        tmp_path = f".uploaded_{up.name}"
        with open(tmp_path, "wb") as f:
            f.write(up.getbuffer())
        df = load_corpus_from_path(tmp_path)
        set_corpus(df, source_label=f"uploaded:{up.name}")
        st.success(f"Loaded {len(df)} rows from upload")

    # Option B: Path or env var
    st.text_input("Path to corpus (optional)", key="corpus_path_text", placeholder="e.g., world_profiles.csv")
    if st.button("Load from path"):
        path = st.session_state.get("corpus_path_text","").strip() or os.environ.get("WORLD_PROFILES_CSV","")
        path = "/Users/sudhirsingh/PyCharmProjects/story/src/services/frontend/world_profiles.csv"
        if not path:
            st.warning("Provide a path or set WORLD_PROFILES_CSV")
        else:
            try:
                df = cache_load_corpus(path)
                set_corpus(df, source_label=f"path:{path}")
                st.success(f"Loaded {len(df)} rows from {path}")
            except Exception as e:
                st.error(f"Failed to load: {e}")

# Guard if corpus not loaded
if "profiles_df" not in st.session_state:
    st.warning("No corpus loaded yet. Upload a file or load from a path in the sidebar.", icon="⚠️")
    st.stop()

# Dataset/corpus badge
meta = st.session_state.get("corpus_meta", {})
if meta:
    st.info(
        f"Corpus: **{meta.get('rows','?')} rows** • Source **{meta.get('source','?')}** • "
        f"Loaded **{meta.get('generated_at','?')}**\n"
        f"Sample IDs: `{', '.join(meta.get('sample_ids', [])[:3])}...`",
        icon="🧪"
    )

# --- Viewer (no-login user switcher) ---
with st.container(border=True):
    st.subheader("Viewer")
    left, mid, right = st.columns([2,2,2])
    with left:
        all_users = list(st.session_state.users.keys())
        if st.session_state.active_user not in all_users and all_users:
            st.session_state.active_user = all_users[0]
        viewer = st.selectbox("Viewing as", all_users, index=all_users.index(st.session_state.active_user))
        st.session_state.active_user = viewer
    with mid:
        new_name = st.text_input("Create new viewer", placeholder="e.g., Neha")
        if st.button("➕ Add Viewer") and new_name:
            if new_name in st.session_state.users:
                st.warning("That name already exists.")
            else:
                st.session_state.users[new_name] = {
                    "settings": {
                        "name": new_name, "age": 28,
                        # City dropdown will be based on the loaded corpus below
                        "city": "",
                        "seeking": ["Woman","Man"],
                        "age_min": 22, "age_max": 40,
                        "top_interests": ["Music","Travel","Foodie"],
                        "weights": {"age": 0.3, "distance": 0.2, "interests": 0.5},
                    },
                    "likes": [], "passes": [], "superlikes": [],
                    "current_index": 0,
                }
                st.session_state.active_user = new_name
                st.rerun()
    with right:
        if st.button("🗑️ Reset this viewer's decisions"):
            ustate = get_active()
            ustate["likes"].clear()
            ustate["passes"].clear()
            ustate["superlikes"].clear()
            ustate["current_index"] = 0
            st.success("Viewer decisions reset.")

# --- Sidebar: per-user settings (dataset-aware) ---
with st.sidebar:
    st.header("Your Settings (for this viewer)")
    ustate = get_active()
    s = ustate["settings"]

    df_all = st.session_state.profiles_df
    # Build available cities dynamically from corpus
    dataset_cities = sorted(df_all["city"].dropna().unique().tolist()) if "city" in df_all.columns else []
    default_city = s.get("city") if s.get("city") in dataset_cities else (dataset_cities[0] if dataset_cities else "")

    s["name"] = st.text_input("Your name", s.get("name",""))
    s["age"] = st.number_input("Your age", min_value=18, max_value=80, value=int(s.get("age",28)), step=1)
    s["city"] = st.selectbox("Your city", dataset_cities or [""], index=(dataset_cities.index(default_city) if dataset_cities and default_city in dataset_cities else 0))
    s["seeking"] = st.multiselect("Show me", GENDERS, default=s.get("seeking", ["Woman","Man"]))
    c1, c2 = st.columns(2)
    with c1:
        s["age_min"] = st.number_input("Min age", 18, 80, int(s.get("age_min",22)), step=1)
    with c2:
        s["age_max"] = st.number_input("Max age", 18, 80, int(s.get("age_max",40)), step=1)

    # Interest universe drawn from data if present
    if "interests" in df_all.columns:
        inferred = sorted({i for L in df_all["interests"] if isinstance(L, list) for i in L})
        universe = inferred or ALL_INTERESTS
    else:
        universe = ALL_INTERESTS
    st.markdown("**Top interests** (helps ranking)")
    default_interests = [i for i in s.get("top_interests", []) if i in universe] or universe[:3]
    s["top_interests"] = st.multiselect("Pick up to 5", universe, default=default_interests, max_selections=5)

    st.markdown("**Scoring weights**")
    age_w = st.slider("Age fit", 0.0, 1.0, float(s["weights"]["age"]), 0.05)
    dist_w = st.slider("Distance", 0.0, 1.0, float(s["weights"]["distance"]), 0.05)
    int_w = st.slider("Interests overlap", 0.0, 1.0, float(s["weights"]["interests"]), 0.05)
    total = age_w + dist_w + int_w or 1.0
    s["weights"] = {"age": age_w/total, "distance": dist_w/total, "interests": int_w/total}

sort_by = st.selectbox("Sort by", ["Best match", "Nearest", "Shuffle"], index=0)

df_ranked = filtered_sorted_profiles(st.session_state.profiles_df, get_active()["settings"], sort_by=sort_by)

# Overall stats row (per viewer)
m1, m2, m3, m4 = st.columns(4)
m1.metric("Profiles available", len(df_ranked) if df_ranked is not None else 0)
m2.metric("Likes", len(get_active()["likes"]))
m3.metric("Superlikes", len(get_active()["superlikes"]))
m4.metric("Passes", len(get_active()["passes"]))

tabs = st.tabs(["Browse", "Grid", "Likes & Passes", "Debug"])

with tabs[0]:
    st.subheader("Swipe-ish")
    if df_ranked is None or df_ranked.empty:
        st.info("No profiles match current filters yet.")
    else:
        idx = get_active()["current_index"]
        if idx >= len(df_ranked):
            st.success("You're all caught up! Adjust filters or load a larger corpus.")
        else:
            row = df_ranked.iloc[idx]
            profile_card(row)
            action_bar(row, get_active())

with tabs[1]:
    st.subheader("All Profiles")
    if df_ranked is None or df_ranked.empty:
        st.info("Nothing to show. Load a corpus and/or relax filters.")
    else:
        n_cols = 3
        rows = [df_ranked.iloc[i:i+n_cols] for i in range(0, len(df_ranked), n_cols)]
        for chunk in rows:
            cols = st.columns(n_cols)
            for col, (_, r) in zip(cols, chunk.iterrows()):
                with col:
                    with st.container(border=True):
                        st.image(r.get("photo_url",""), use_container_width=True)
                        st.write(f"**{r.get('name','')}**, {r.get('age','—')} • {r.get('gender','')}")
                        city = r.get('city',''); dist = r.get('distance_km','—'); comp = r.get('compatibility',0.0)
                        st.caption(f"📍 {city} • ~{dist} km • Compat {comp:.2f}")
                        ints = r.get("interests", [])
                        if isinstance(ints, list):
                            st.caption(", ".join(ints))
                        c1, c2 = st.columns(2)
                        with c1:
                            if st.button("❤️", key=f"grid_like_{st.session_state.active_user}_{r['id']}"):
                                get_active()["likes"].append(r["id"])
                        with c2:
                            if st.button("👎", key=f"grid_pass_{st.session_state.active_user}_{r['id']}"):
                                get_active()["passes"].append(r["id"])

with tabs[2]:
    st.subheader("Your Decisions")
    ustate = get_active()
    if df_ranked is None or df_ranked.empty:
        st.info("No ranked dataset yet.")
    else:
        liked_ids = set(ustate["likes"] + ustate["superlikes"])
        liked_df = df_ranked[df_ranked["id"].isin(liked_ids)] if "id" in df_ranked.columns else pd.DataFrame()
        passed_df = df_ranked[df_ranked["id"].isin(ustate["passes"])] if "id" in df_ranked.columns else pd.DataFrame()
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("### ❤️ Likes & ⭐ Superlikes")
            for _, r in liked_df.iterrows():
                with st.container(border=True):
                    st.write(f"**{r.get('name','')}**, {r.get('age','—')} • {r.get('gender','')} — Compat {r.get('compatibility',0.0):.2f}")
                    st.caption(f"📍 {r.get('city','')} • ~{r.get('distance_km','—')} km")
        with c2:
            st.markdown("### 👎 Passes")
            for _, r in passed_df.iterrows():
                with st.container(border=True):
                    st.write(f"**{r.get('name','')}**, {r.get('age','—')} • {r.get('gender','')} — Compat {r.get('compatibility',0.0):.2f}")
                    st.caption(f"📍 {r.get('city','')} • ~{r.get('distance_km','—')} km")

        st.divider()
        export_buttons(st.session_state.profiles_df, st.session_state.active_user, ustate)

with tabs[3]:
    st.subheader("Debug / Developer Hooks")
    st.write("**Active viewer settings**")
    st.json(get_active()["settings"])
    st.write("**Corpus meta**")
    st.json(st.session_state.get("corpus_meta", {}))
    st.write("**Current dataset (ranked for this viewer)**")
    st.dataframe(df_ranked, use_container_width=True)

    st.info(
        "Plug your ML recommender here later. Just replace `compute_compatibility` with your model's score,\n"
        "or add a new column and sort by it."
    )
