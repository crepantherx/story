
# app.py — Minimal Dating App Prototype (Multi-User "View As" Edition)
# Run with: streamlit run app.py

import streamlit as st
import pandas as pd
import numpy as np
import random
import uuid
from datetime import datetime

# ----------------------------
# Page config
# ----------------------------
st.set_page_config(
    page_title="Prototype",
    page_icon="💘",
    layout="wide",
)

# ----------------------------
# Constants
# ----------------------------
RNG = random.Random(42)

CITIES = [
    "Mumbai", "Delhi", "Bengaluru", "Hyderabad", "Chennai",
    "Pune", "Kolkata", "Ahmedabad", "Jaipur", "Surat"
]

GENDERS = ["Woman", "Man", "Non-binary"]
INTERESTS = [
    "Hiking", "Cooking", "Reading", "Music", "Movies", "Art",
    "Gaming", "Travel", "Yoga", "Running", "Photography", "Dancing",
    "Tech", "Foodie", "Startups", "Cricket", "Football", "Basketball",
    "Poetry", "Theatre", "Standup Comedy", "Board Games", "Volunteering"
]

DEMO_USERS = {
    "Aditi": {
        "name": "Aditi", "age": 27, "city": "Mumbai",
        "seeking": ["Man"],
        "age_min": 24, "age_max": 34,
        "top_interests": ["Music", "Travel", "Foodie"],
        "weights": {"age": 0.3, "distance": 0.2, "interests": 0.5},
    },
    "Rahul": {
        "name": "Rahul", "age": 30, "city": "Bengaluru",
        "seeking": ["Woman"],
        "age_min": 23, "age_max": 33,
        "top_interests": ["Tech", "Startups", "Cricket"],
        "weights": {"age": 0.25, "distance": 0.25, "interests": 0.5},
    },
    "Sam": {
        "name": "Sam", "age": 29, "city": "Pune",
        "seeking": ["Woman","Non-binary","Man"],
        "age_min": 22, "age_max": 40,
        "top_interests": ["Art", "Theatre", "Poetry"],
        "weights": {"age": 0.2, "distance": 0.4, "interests": 0.4},
    },
}

# ----------------------------
# Fake data helpers
# ----------------------------
def random_interests(k=None):
    k = k or RNG.randint(3, 6)
    return RNG.sample(INTERESTS, k)


def make_fake_profiles(n=10, seed=123):
    rng = random.Random(seed)
    rows = []
    for i in range(n):
        pid = str(uuid.uuid4())[:8]
        age = rng.randint(21, 45)
        gender = rng.choice(GENDERS)
        city = rng.choice(CITIES)
        km = rng.randint(1, 30)  # fake distance
        about = rng.choice([
            "Looking for coffee buddies and spontaneous day trips.",
            "Bookworm, brunch enthusiast, weekend trekker.",
            "Techie by day, home chef by night.",
            "Gym sometimes, dessert always.",
            "Will trade Spotify playlists for food recs.",
            "Here for the memes and the banter.",
            "Plant parent. Can keep succulents alive (usually).",
            "Learning guitar, send tips (and patience).",
        ])
        photo_url = f"https://picsum.photos/seed/{pid}/480/480"
        interests = random_interests()
        rows.append({
            "id": pid,
            "name": rng.choice(["Aarav","Vivaan","Aditya","Vihaan","Arjun","Sai","Krishna","Ishaan","Rohan","Kabir",
                                "Aadhya","Aarohi","Anaya","Diya","Isha","Myra","Sara","Siya","Tara","Zara"]),
            "age": age,
            "gender": gender,
            "city": city,
            "distance_km": km,
            "about": about,
            "interests": interests,
            "photo_url": photo_url,
        })
    return pd.DataFrame(rows)

# ----------------------------
# Session state scaffolding
# ----------------------------
def ensure_state():
    if "profiles_df" not in st.session_state:
        st.session_state.profiles_df = make_fake_profiles(10, seed=123)
    if "seed" not in st.session_state:
        st.session_state.seed = 123
    if "users" not in st.session_state:
        st.session_state.users = {}
    if "active_user" not in st.session_state:
        # Bootstrap with demo users; default active is Rahul
        for u, settings in DEMO_USERS.items():
            st.session_state.users[u] = {
                "settings": settings.copy(),
                "likes": [], "passes": [], "superlikes": [],
                "current_index": 0,
            }
        st.session_state.active_user = "Rahul"


def get_active():
    return st.session_state.users[st.session_state.active_user]


# ----------------------------
# Matching logic
# ----------------------------
def compute_compatibility(row, settings):
    # Toy score in [0,1]
    w = settings["weights"]
    # Age score
    target_min = settings["age_min"]
    target_max = settings["age_max"]
    if row["age"] < target_min or row["age"] > target_max:
        age_score = 0.0
    else:
        mid = (target_min + target_max) / 2.0
        spread = max((target_max - target_min) / 2.0, 1.0)
        age_score = max(0.0, 1.0 - abs(row["age"] - mid) / spread)

    # Distance: 0–30 km -> 1–0
    dist = row["distance_km"]
    distance_score = max(0.0, 1.0 - (dist / 30.0))

    # Interests overlap
    your_interests = set(settings["top_interests"])
    their_interests = set(row["interests"])
    overlap = (len(your_interests & their_interests) / len(your_interests)) if your_interests else 0.0

    score = w["age"] * age_score + w["distance"] * distance_score + w["interests"] * overlap
    return round(float(score), 3)


def filtered_sorted_profiles(df, settings, sort_by="Best match"):
    # Filter by gender and age
    mask = df["gender"].isin(settings["seeking"]) & df["age"].between(settings["age_min"], settings["age_max"])
    filtered = df[mask].copy()
    # Compute compatibility
    filtered["compatibility"] = filtered.apply(lambda r: compute_compatibility(r, settings), axis=1)
    if sort_by == "Best match":
        filtered = filtered.sort_values(by=["compatibility", "distance_km"], ascending=[False, True])
    elif sort_by == "Nearest":
        filtered = filtered.sort_values(by=["distance_km", "compatibility"], ascending=[True, False])
    else:
        filtered = filtered.sample(frac=1, random_state=42)
    return filtered.reset_index(drop=True)


def profile_card(row):
    # Visual card
    with st.container(border=True):
        c1, c2 = st.columns([1, 2], gap="large")
        with c1:
            st.image(row["photo_url"], use_column_width=True, caption=f"{row['name']}, {row['age']} • {row['gender']}")
            st.caption(f"📍 {row['city']} • ~{row['distance_km']} km away")
            st.progress(row.get("compatibility", 0.0), text=f"Compat: {row.get('compatibility', 0.0):.2f}")
        with c2:
            st.subheader(f"{row['name']}")
            st.write(row["about"])
            st.write("**Interests**:", ", ".join(row["interests"]))


def action_bar(row, user_state):
    c1, c2, c3, c4 = st.columns([1,1,1,6])
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
    # Export likes/passes with profiles joined
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
st.caption("Switch the viewer to see the site from any user's perspective — no login required.")

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
                # Initialize with generic defaults
                st.session_state.users[new_name] = {
                    "settings": {
                        "name": new_name, "age": 28, "city": "Bengaluru",
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

# --- Sidebar: per-user settings ---
with st.sidebar:
    st.header("Your Settings (for this viewer)")
    ustate = get_active()
    s = ustate["settings"]
    s["name"] = st.text_input("Your name", s["name"])
    s["age"] = st.number_input("Your age", min_value=18, max_value=80, value=int(s["age"]), step=1)
    s["city"] = st.selectbox("Your city", CITIES, index=CITIES.index(s["city"]) if s["city"] in CITIES else 2)
    s["seeking"] = st.multiselect("Show me", GENDERS, default=s["seeking"])
    c1, c2 = st.columns(2)
    with c1:
        s["age_min"] = st.number_input("Min age", 18, 80, int(s["age_min"]), step=1)
    with c2:
        s["age_max"] = st.number_input("Max age", 18, 80, int(s["age_max"]), step=1)

    st.markdown("**Top interests** (helps ranking)")
    s["top_interests"] = st.multiselect("Pick up to 5", INTERESTS, default=s["top_interests"], max_selections=5)

    st.markdown("**Scoring weights**")
    age_w = st.slider("Age fit", 0.0, 1.0, float(s["weights"]["age"]), 0.05)
    dist_w = st.slider("Distance", 0.0, 1.0, float(s["weights"]["distance"]), 0.05)
    int_w = st.slider("Interests overlap", 0.0, 1.0, float(s["weights"]["interests"]), 0.05)
    total = age_w + dist_w + int_w or 1.0
    s["weights"] = {"age": age_w/total, "distance": dist_w/total, "interests": int_w/total}

    st.divider()
    st.subheader("Dataset Controls (shared)")
    st.caption("Regenerate the fake dataset with a different seed (affects all viewers).")
    st.session_state.seed = st.number_input("Profiles random seed", 0, 10_000, int(st.session_state.seed), step=1)
    if st.button("🔁 Regenerate 10 fake profiles"):
        st.session_state.profiles_df = make_fake_profiles(10, seed=int(st.session_state.seed))
        # keep each user's current_index reasonable
        for uname in st.session_state.users:
            st.session_state.users[uname]["current_index"] = 0

sort_by = st.selectbox("Sort by", ["Best match", "Nearest", "Shuffle"], index=0)

df = filtered_sorted_profiles(st.session_state.profiles_df, get_active()["settings"], sort_by=sort_by)

# Overall stats row (per viewer)
m1, m2, m3, m4 = st.columns(4)
m1.metric("Profiles available", len(df))
m2.metric("Likes", len(get_active()["likes"]))
m3.metric("Superlikes", len(get_active()["superlikes"]))
m4.metric("Passes", len(get_active()["passes"]))

tabs = st.tabs(["Browse", "Grid", "Likes & Passes", "Debug"])

with tabs[0]:
    st.subheader("Swipe-ish")
    idx = get_active()["current_index"]
    if idx >= len(df):
        st.success("You're all caught up! Adjust filters or regenerate profiles.")
    else:
        row = df.iloc[idx]
        profile_card(row)
        action_bar(row, get_active())

with tabs[1]:
    st.subheader("All Profiles")
    # Grid view of cards
    n_cols = 3
    rows = [df.iloc[i:i+n_cols] for i in range(0, len(df), n_cols)]
    for chunk in rows:
        cols = st.columns(n_cols)
        for col, (_, r) in zip(cols, chunk.iterrows()):
            with col:
                with st.container(border=True):
                    st.image(r["photo_url"], use_column_width=True)
                    st.write(f"**{r['name']}**, {r['age']} • {r['gender']}")
                    st.caption(f"📍 {r['city']} • ~{r['distance_km']} km • Compat {r['compatibility']:.2f}")
                    st.caption(", ".join(r["interests"]))
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
    liked_ids = set(ustate["likes"] + ustate["superlikes"])
    liked_df = df[df["id"].isin(liked_ids)]
    passed_df = df[df["id"].isin(ustate["passes"])]
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### ❤️ Likes & ⭐ Superlikes")
        for _, r in liked_df.iterrows():
            with st.container(border=True):
                st.write(f"**{r['name']}**, {r['age']} • {r['gender']} — Compat {r['compatibility']:.2f}")
                st.caption(f"📍 {r['city']} • ~{r['distance_km']} km")
    with c2:
        st.markdown("### 👎 Passes")
        for _, r in passed_df.iterrows():
            with st.container(border=True):
                st.write(f"**{r['name']}**, {r['age']} • {r['gender']} — Compat {r['compatibility']:.2f}")
                st.caption(f"📍 {r['city']} • ~{r['distance_km']} km")

    st.divider()
    export_buttons(st.session_state.profiles_df, st.session_state.active_user, ustate)

with tabs[3]:
    st.subheader("Debug / Developer Hooks")
    st.write("**Active viewer settings**")
    st.json(get_active()["settings"])
    st.write("**Current dataset (ranked for this viewer)**")
    st.dataframe(df, use_container_width=True)

    st.info(
        "Future work idea: Replace `compute_compatibility` with your trained recommender.\n"
        "The multi-user scaffold lets you compare model behavior for different personas without implementing auth yet."
    )
