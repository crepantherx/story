# app.py — Minimal Dating App Prototype (Multi-User "View As" Edition)
# Run with: streamlit run app.py

import os
import csv
import json
import streamlit as st
import pandas as pd
import numpy as np
import random
import uuid
from datetime import datetime

# ================================================================
# IMPORT/DEFINE: World profile generator (from your earlier snippet)
# (Inlined here for a single-file app.)
# ================================================================
DEFAULT_SEED = 123
rng = random.Random(DEFAULT_SEED)

GENDERS = ["Woman", "Man", "Non-binary"]

INTEREST_CLUSTERS = {
    "Active": ["Hiking", "Running", "Yoga", "Dancing", "Photography"],
    "Arts": ["Art", "Theatre", "Poetry", "Movies", "Music"],
    "Geek": ["Tech", "Gaming", "Startups", "Board Games"],
    "Social": ["Foodie", "Travel", "Standup Comedy", "Volunteering"],
    "Sports": ["Cricket", "Football", "Basketball"],
}
ALL_INTERESTS = sorted({i for v in INTEREST_CLUSTERS.values() for i in v})

FEMALE_FIRST = [
    "Aditi","Aarohi","Anaya","Diya","Isha","Myra","Sara","Siya","Tara","Zara",
    "Neha","Priya","Naina","Rhea","Meera","Anika","Kavya","Ritu","Pooja","Sana",
    "Anna","Maria","Sofia","Emma","Olivia","Mia","Aisha","Fatima","Yuna","Mei",
    "Camila","Valentina","Amara","Zainab","Helena","Elena","Giulia","Lina","Aya"
]
MALE_FIRST = [
    "Aarav","Vivaan","Aditya","Vihaan","Arjun","Sai","Krishna","Ishaan","Rohan","Kabir",
    "Raghav","Aman","Rajat","Varun","Anil","Rahul","Aakash","Nikhil","Sandeep","Yash",
    "Liam","Noah","Lucas","Mateo","Ethan","Leo","Hiro","Daichi","Minjun","Jae",
    "Luis","Diego","Andre","Omar","Youssef","Ali","Marco","Jonas","Felix","Tariq"
]
UNISEX_FIRST = ["Sam","Dev","Shiv","Arya","Sasha","Riyaan","Jai","Ray","Kiran","Alex","Charlie","Noor","Ariel","Jordan","Kai"]

INDIA_TIERS = {
    "Tier-1": [
        ("India", "Mumbai", 10),
        ("India", "Delhi", 10),
        ("India", "Bengaluru", 9),
        ("India", "Hyderabad", 8),
        ("India", "Chennai", 7),
        ("India", "Kolkata", 7),
        ("India", "Pune", 6),
        ("India", "Ahmedabad", 5),
    ],
    "Tier-2": [
        ("India", "Jaipur", 4), ("India", "Surat", 4), ("India", "Lucknow", 4),
        ("India", "Kanpur", 3), ("India", "Nagpur", 3), ("India", "Indore", 3),
        ("India", "Bhopal", 3), ("India", "Chandigarh", 2), ("India", "Kochi", 2),
        ("India", "Coimbatore", 2),
    ],
    "Tier-3": [
        ("India", "Patna", 2), ("India", "Guwahati", 2), ("India", "Visakhapatnam", 2),
        ("India", "Vijayawada", 2), ("India", "Bhubaneswar", 2), ("India", "Thiruvananthapuram", 2),
        ("India", "Vadodara", 2), ("India", "Nashik", 2), ("India", "Ludhiana", 2), ("India", "Rajkot", 2),
    ],
}

WORLD_REGIONS = {
    "South Asia (non-India)": [
        ("Bangladesh", "Dhaka", 8), ("Bangladesh", "Chittagong", 3),
        ("Pakistan", "Karachi", 9), ("Pakistan", "Lahore", 6), ("Pakistan", "Islamabad", 2),
        ("Sri Lanka", "Colombo", 2), ("Nepal", "Kathmandu", 2),
    ],
    "East Asia": [
        ("Japan", "Tokyo", 10), ("Japan", "Osaka", 4),
        ("South Korea", "Seoul", 8), ("South Korea", "Busan", 3),
        ("China", "Shanghai", 10), ("China", "Beijing", 9), ("China", "Shenzhen", 7), ("China", "Guangzhou", 7),
        ("Taiwan", "Taipei", 4), ("Hong Kong", "Hong Kong", 5),
    ],
    "Southeast Asia": [
        ("Singapore", "Singapore", 6),
        ("Malaysia", "Kuala Lumpur", 4),
        ("Thailand", "Bangkok", 7),
        ("Indonesia", "Jakarta", 9), ("Vietnam", "Ho Chi Minh City", 6), ("Vietnam", "Hanoi", 5),
        ("Philippines", "Manila", 8),
    ],
    "North America": [
        ("USA", "New York", 9), ("USA", "Los Angeles", 8), ("USA", "Chicago", 6),
        ("USA", "San Francisco", 5), ("USA", "Houston", 5), ("USA", "Miami", 5),
        ("Canada", "Toronto", 6), ("Canada", "Vancouver", 4), ("Canada", "Montreal", 4),
        ("Mexico", "Mexico City", 9), ("Mexico", "Guadalajara", 4),
    ],
    "Europe": [
        ("UK", "London", 9), ("France", "Paris", 8), ("Germany", "Berlin", 6),
        ("Spain", "Madrid", 5), ("Spain", "Barcelona", 5),
        ("Italy", "Rome", 5), ("Italy", "Milan", 4),
        ("Netherlands", "Amsterdam", 4), ("Austria", "Vienna", 4), ("Sweden", "Stockholm", 3),
    ],
    "MENA": [
        ("UAE", "Dubai", 7), ("UAE", "Abu Dhabi", 4),
        ("Saudi Arabia", "Riyadh", 6), ("Saudi Arabia", "Jeddah", 5),
        ("Egypt", "Cairo", 8), ("Egypt", "Alexandria", 4),
        ("Türkiye", "Istanbul", 8), ("Morocco", "Casablanca", 4),
    ],
    "Sub-Saharan Africa": [
        ("Nigeria", "Lagos", 9), ("Nigeria", "Abuja", 4),
        ("Kenya", "Nairobi", 6), ("Kenya", "Mombasa", 3),
        ("Ghana", "Accra", 4), ("Ghana", "Kumasi", 3),
        ("South Africa", "Johannesburg", 5), ("South Africa", "Cape Town", 5), ("South Africa", "Durban", 3),
        ("Ethiopia", "Addis Ababa", 5),
    ],
    "Latin America": [
        ("Brazil", "São Paulo", 10), ("Brazil", "Rio de Janeiro", 7),
        ("Argentina", "Buenos Aires", 8), ("Chile", "Santiago", 6),
        ("Peru", "Lima", 7), ("Colombia", "Bogotá", 7), ("Colombia", "Medellín", 4),
        ("Ecuador", "Quito", 3), ("Uruguay", "Montevideo", 3),
    ],
    "Oceania": [
        ("Australia", "Sydney", 6), ("Australia", "Melbourne", 6),
        ("Australia", "Brisbane", 3), ("Australia", "Perth", 3),
        ("New Zealand", "Auckland", 3), ("New Zealand", "Wellington", 2),
    ],
}

def build_city_table(include_india=True, india_tier_bias=(0.5, 0.35, 0.15)):
    rows = []
    if include_india:
        tiers = ["Tier-1", "Tier-2", "Tier-3"]
        tier_w = dict(zip(tiers, india_tier_bias))
        for tier in tiers:
            for country, city, w in INDIA_TIERS[tier]:
                rows.append(("South Asia", country, city, w * (1 + 9 * tier_w[tier])))
    for region, cities in WORLD_REGIONS.items():
        for country, city, w in cities:
            rows.append((region, country, city, w))
    return rows

WORLD_CITY_TABLE = build_city_table()

def randomuser_url(pid: str, gender: str) -> str:
    idx = int(pid, 16) % 100
    if gender == "Woman":
        folder = "women"
    elif gender == "Man":
        folder = "men"
    else:
        folder = "women" if (idx % 2 == 0) else "men"
    return f"https://randomuser.me/api/portraits/{folder}/{idx}.jpg"

def sample_world_city():
    weights = [w for (_, _, _, w) in WORLD_CITY_TABLE]
    choices = [(r, ctry, cty) for (r, ctry, cty, _) in WORLD_CITY_TABLE]
    return rng.choices(choices, weights=weights, k=1)[0]

def sample_gender():
    return rng.choices(GENDERS, weights=[0.47, 0.47, 0.06], k=1)[0]

def sample_name(gender):
    if gender == "Woman":
        pool = FEMALE_FIRST + UNISEX_FIRST
    elif gender == "Man":
        pool = MALE_FIRST + UNISEX_FIRST
    else:
        pool = UNISEX_FIRST + FEMALE_FIRST[:10] + MALE_FIRST[:10]
    return rng.choice(pool)

def truncated_normal(mean, sd, lo, hi):
    while True:
        x = rng.gauss(mean, sd)
        if lo <= x <= hi:
            return int(round(x))

def sample_age(region):
    mean = 27
    if region in {"Europe","North America"}: mean = 29
    if region in {"South Asia","South Asia (non-India)","Africa","Sub-Saharan Africa"}: mean = 26
    return truncated_normal(mean, 4.5, 21, 45)

def sample_distance_km(region):
    lam = 1 / 6.0
    val = int(round(min(30, max(1, rng.expovariate(lam)))))
    if region in {"Europe","North America"} and rng.random() < 0.25:
        val = min(30, val + rng.randint(2,5))
    return val

def pick_interest_cluster(age, region):
    w = {"Active":1,"Arts":1,"Geek":1,"Social":1,"Sports":1}
    if age <= 26: w["Geek"] += 0.6; w["Social"] += 0.4
    if age >= 30: w["Arts"] += 0.4; w["Active"] += 0.2
    if region in {"Europe","North America"}: w["Arts"] += 0.2
    if region in {"South Asia","South Asia (non-India)","East Asia"}: w["Geek"] += 0.3
    keys = list(INTEREST_CLUSTERS.keys())
    return rng.choices(keys, weights=[w[k] for k in keys], k=1)[0]

def sample_interests(age, region):
    k = rng.randint(3,6)
    base = pick_interest_cluster(age, region)
    alt = base if rng.random() < 0.6 else rng.choice(list(INTEREST_CLUSTERS.keys()))
    pool = list(dict.fromkeys(INTEREST_CLUSTERS[base] + INTEREST_CLUSTERS[alt]))
    if rng.random() < 0.35:
        extras = [i for i in ALL_INTERESTS if i not in pool]
        if extras:
            pool += rng.sample(extras, k=min(3, len(extras)))
    rng.shuffle(pool)
    return pool[:k]

def make_bio(name, age, city, interests):
    lead = rng.choice([
        "Powered by coffee and chaotic good energy.",
        "Part-time explorer, full-time snack enthusiast.",
        "Weekends = long walks + long playlists.",
        "Trying new things and new foods—recommendations welcome.",
        "Recovering overthinker, thriving bruncher.",
        "Swaps memes for restaurant tips.",
    ])
    hook = rng.choice([
        f"Into {interests[0].lower()} and {interests[1].lower()}",
        f"{interests[0]} > {interests[1]}? Discuss.",
        f"If you like {interests[0].lower()}, we’ll get along.",
        f"From {city}, chasing {interests[-1].lower()} vibes.",
        f"{interests[0]}, {interests[1]}, and probably {interests[-1].lower()}",
    ])
    closer = rng.choice([
        "Coffee then a walk?",
        "Open to spontaneous day trips.",
        "Here for good banter and better food.",
        "Teach me your niche skill.",
        "Playlist swaps encouraged.",
    ])
    return f"{lead} {hook}. {closer}"

def make_world_profiles(n=10000, seed=DEFAULT_SEED):
    rng.seed(seed)
    rows = []
    for _ in range(n):
        pid = str(uuid.uuid4())[:8]
        region, country, city = sample_world_city()
        gender = sample_gender()
        name = sample_name(gender)
        age = sample_age(region)
        distance = sample_distance_km(region)
        interests = sample_interests(age, region)
        bio = make_bio(name, age, city, interests)
        photo_url = randomuser_url(pid, gender)

        rows.append({
            "id": pid,
            "name": name,
            "age": age,
            "gender": gender,
            "region": region,
            "country": country,
            "city": city,
            "distance_km": distance,
            "interests": interests,
            "about": bio,
            "photo_url": photo_url,
        })
    return pd.DataFrame(rows)

# ----------------------------
# Original demo users
# ----------------------------
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
# Page config
# ----------------------------
st.set_page_config(
    page_title="App Prototype",
    page_icon="💘",
    layout="wide",
)

# ----------------------------
# Viewers persistence helpers (Option 1)
# ----------------------------
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

# ----------------------------
# Session state scaffolding
# ----------------------------
def ensure_state():
    if "profiles_df" not in st.session_state:
        st.session_state.profiles_df = make_world_profiles(n=200, seed=123)
    if "seed" not in st.session_state:
        st.session_state.seed = 123
    if "size" not in st.session_state:
        st.session_state.size = 200
    if "users" not in st.session_state:
        st.session_state.users = {}
    if "active_user" not in st.session_state:
        for u, settings in DEMO_USERS.items():
            st.session_state.users[u] = {
                "settings": settings.copy(),
                "likes": [], "passes": [], "superlikes": [],
                "current_index": 0,
            }
        st.session_state.active_user = "Rahul"
    if "interactions_csv" not in st.session_state:
        st.session_state.interactions_csv = os.environ.get("INTERACTIONS_CSV", "interactions_log.csv")
    if "viewers_csv" not in st.session_state:
        st.session_state.viewers_csv = os.environ.get("VIEWERS_CSV", "viewers.csv")

def get_active():
    return st.session_state.users[st.session_state.active_user]

# ----------------------------
# Interaction logging
# ----------------------------
INTERACTION_FIELDS = ["timestamp","viewer_id","viewer_name","profile_id","profile_name","action","compatibility"]

def log_interaction(viewer_key: str, viewer_name: str, profile_row: pd.Series, action: str, compatibility: float):
    path = st.session_state.get("interactions_csv", "interactions_log.csv")
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

# ----------------------------
# Matching logic (unchanged)
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

    dist = row["distance_km"]
    distance_score = max(0.0, 1.0 - (dist / 30.0))

    your_interests = set(st.session_state.users[st.session_state.active_user]["settings"]["top_interests"])
    their_interests = set(row["interests"])
    overlap = (len(your_interests & their_interests) / len(your_interests)) if your_interests else 0.0

    score = w["age"] * age_score + w["distance"] * distance_score + w["interests"] * overlap
    return round(float(score), 3)

def filtered_sorted_profiles(df, settings, sort_by="Best match"):
    mask = df["gender"].isin(settings["seeking"]) & df["age"].between(settings["age_min"], settings["age_max"])
    filtered = df[mask].copy()
    filtered["compatibility"] = filtered.apply(lambda r: compute_compatibility(r, settings), axis=1)
    if sort_by == "Best match":
        filtered = filtered.sort_values(by=["compatibility", "distance_km"], ascending=[False, True])
    elif sort_by == "Nearest":
        filtered = filtered.sort_values(by=["distance_km", "compatibility"], ascending=[True, False])
    else:
        filtered = filtered.sample(frac=1, random_state=42)
    return filtered.reset_index(drop=True)

def profile_card(row):
    with st.container(border=True):
        c1, c2 = st.columns([1, 2], gap="large")
        with c1:
            st.image(row["photo_url"], use_container_width=True, caption=f"{row['name']}, {row['age']} • {row['gender']}")
            st.caption(f"📍 {row['city']} • ~{row['distance_km']} km away")
            st.progress(row.get("compatibility", 0.0), text=f"Compat: {row.get('compatibility', 0.0):.2f}")
        with c2:
            st.subheader(f"{row['name']}")
            st.write(row["about"])
            st.write("**Interests**:", ", ".join(row["interests"]))

def action_bar(row, user_state):
    c1, c2, c3, _ = st.columns([1,1,1,6])
    with c1:
        if st.button("👎 Pass", key=f"pass_{row['id']}"):
            user_state["passes"].append(row["id"])
            log_interaction(
                viewer_key=st.session_state.active_user,
                viewer_name=user_state["settings"]["name"],
                profile_row=row,
                action="pass",
                compatibility=row.get("compatibility", 0.0),
            )
            user_state["current_index"] += 1
            st.rerun()
    with c2:
        if st.button("⭐ Superlike", key=f"super_{row['id']}"):
            user_state["superlikes"].append(row["id"])
            log_interaction(
                viewer_key=st.session_state.active_user,
                viewer_name=user_state["settings"]["name"],
                profile_row=row,
                action="superlike",
                compatibility=row.get("compatibility", 0.0),
            )
            user_state["current_index"] += 1
            st.rerun()
    with c3:
        if st.button("❤️ Like", key=f"like_{row['id']}"):
            user_state["likes"].append(row["id"])
            log_interaction(
                viewer_key=st.session_state.active_user,
                viewer_name=user_state["settings"]["name"],
                profile_row=row,
                action="like",
                compatibility=row.get("compatibility", 0.0),
            )
            user_state["current_index"] += 1
            st.rerun()

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

# ----------------------------
# App
# ----------------------------
ensure_state()

st.title("Recommendation")
st.caption("Switch the viewer to see the site from any user's perspective — log interactions and persist viewers.")

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
        new_name = st.text_input("Create new viewer (manual)", placeholder="e.g., Neha")
        if st.button("➕ Add Viewer") and new_name:
            if new_name in st.session_state.users:
                st.warning("That name already exists.")
            else:
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
                # Persist this new viewer
                upsert_viewer(
                    st.session_state.users[new_name]["settings"],
                    viewer_id=new_name,
                    path=st.session_state.viewers_csv,
                )
                st.session_state.active_user = new_name
                st.rerun()
    with right:
        st.markdown("**Make viewer from a profile**")
        df_choices = st.session_state.profiles_df
        choices = [f"{r['name']} ({r['id']}) — {r['city']}, {r['country']}" for _, r in df_choices.iterrows()]
        choice_ix = st.selectbox("Pick profile", options=list(range(len(choices))), format_func=lambda i: choices[i] if choices else "—")
        if st.button("👤 Use this profile as viewer"):
            pr = df_choices.iloc[choice_ix]
            vname = f"{pr['name']}-{pr['id']}"
            st.session_state.users[vname] = {
                "settings": {
                    "name": pr["name"],
                    "age": int(pr["age"]),
                    "city": pr.get("city", ""),
                    "seeking": GENDERS[:],
                    "age_min": max(18, int(pr["age"]) - 5),
                    "age_max": min(80, int(pr["age"]) + 5),
                    "top_interests": list(pr.get("interests", [])[:3]) if isinstance(pr.get("interests", []), list) else [],
                    "weights": {"age": 0.3, "distance": 0.2, "interests": 0.5},
                },
                "likes": [], "passes": [], "superlikes": [],
                "current_index": 0,
            }
            # Persist this viewer
            upsert_viewer(
                st.session_state.users[vname]["settings"],
                viewer_id=vname,
                path=st.session_state.viewers_csv,
            )
            st.session_state.active_user = vname
            st.success(f"Now viewing as {vname}")
            st.rerun()

# --- Sidebar: per-user settings + dataset controls + logging/persistence paths ---
with st.sidebar:
    st.header("Your Settings (for this viewer)")
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

    generator_interests = sorted(ALL_INTERESTS)
    st.markdown("**Top interests** (helps ranking)")
    s["top_interests"] = st.multiselect("Pick up to 5", generator_interests, default=[i for i in s["top_interests"] if i in generator_interests] or generator_interests[:3], max_selections=5)

    st.markdown("**Scoring weights**")
    age_w = st.slider("Age fit", 0.0, 1.0, float(s["weights"]["age"]), 0.05)
    dist_w = st.slider("Distance", 0.0, 1.0, float(s["weights"]["distance"]), 0.05)
    int_w = st.slider("Interests overlap", 0.0, 1.0, float(s["weights"]["interests"]), 0.05)
    total = age_w + dist_w + int_w or 1.0
    s["weights"] = {"age": age_w/total, "distance": dist_w/total, "interests": int_w/total}

    # Persist viewer after any changes
    upsert_viewer(
        settings=s,
        viewer_id=st.session_state.active_user,
        path=st.session_state.viewers_csv,
    )

    st.divider()
    st.subheader("Dataset Controls (shared)")
    st.caption("Regenerate the synthetic dataset using your world-profile generator.")
    st.session_state.seed = st.number_input("Random seed", 0, 1_000_000, int(st.session_state.seed), step=1)
    st.session_state.size = st.number_input("Number of profiles", 50, 50_000, int(st.session_state.size), step=50)
    if st.button("🔁 Regenerate dataset"):
        st.session_state.profiles_df = make_world_profiles(n=int(st.session_state.size), seed=int(st.session_state.seed))
        for uname in st.session_state.users:
            st.session_state.users[uname]["current_index"] = 0
        st.success(f"Regenerated: {len(st.session_state.profiles_df)} profiles")

    st.divider()
    st.subheader("Logging")
    st.text_input("Interactions CSV path", key="interactions_csv", value=st.session_state.get("interactions_csv", "interactions_log.csv"))
    if os.path.exists(st.session_state.interactions_csv):
        st.caption(f"Logging to: `{st.session_state.interactions_csv}` (exists)")
    else:
        st.caption(f"Will create: `{st.session_state.interactions_csv}`")

    st.divider()
    st.subheader("Personas (Viewers) persistence")
    st.text_input("Viewers CSV path", key="viewers_csv", value=st.session_state.get("viewers_csv", "viewers.csv"))
    if os.path.exists(st.session_state.viewers_csv):
        st.caption(f"Persisting viewers to: `{st.session_state.viewers_csv}` (exists)")
    else:
        st.caption(f"Will create: `{st.session_state.viewers_csv}` on first save")
    if st.button("👀 Show saved viewers.csv"):
        st.dataframe(_load_viewers_df(st.session_state.viewers_csv), use_container_width=True)

sort_by = st.selectbox("Sort by", ["Best match", "Nearest", "Shuffle"], index=0)

df_ranked = filtered_sorted_profiles(st.session_state.profiles_df, get_active()["settings"], sort_by=sort_by)

# Overall stats row (per viewer)
m1, m2, m3, m4 = st.columns(4)
m1.metric("Profiles available", len(df_ranked))
m2.metric("Likes", len(get_active()["likes"]))
m3.metric("Superlikes", len(get_active()["superlikes"]))
m4.metric("Passes", len(get_active()["passes"]))

tabs = st.tabs(["Browse", "Grid", "Likes & Passes", "Debug"])

with tabs[0]:
    st.subheader("Swipe-ish")
    idx = get_active()["current_index"]
    if idx >= len(df_ranked):
        st.success("You're all caught up! Adjust filters or regenerate profiles.")
    else:
        row = df_ranked.iloc[idx]
        profile_card(row)
        action_bar(row, get_active())

with tabs[1]:
    st.subheader("All Profiles")
    n_cols = 3
    rows = [df_ranked.iloc[i:i+n_cols] for i in range(0, len(df_ranked), n_cols)]
    for chunk in rows:
        cols = st.columns(n_cols)
        for col, (_, r) in zip(cols, chunk.iterrows()):
            with col:
                with st.container(border=True):
                    st.image(r["photo_url"], use_container_width=True)
                    st.write(f"**{r['name']}**, {r['age']} • {r['gender']}")
                    st.caption(f"📍 {r['city']} • ~{r['distance_km']} km • Compat {r['compatibility']:.2f}")
                    st.caption(", ".join(r["interests"]))
                    c1, c2 = st.columns(2)
                    with c1:
                        if st.button("❤️", key=f"grid_like_{st.session_state.active_user}_{r['id']}"):
                            get_active()["likes"].append(r["id"])
                            log_interaction(st.session_state.active_user, get_active()["settings"]["name"], r, "like", r.get("compatibility", 0.0))
                    with c2:
                        if st.button("👎", key=f"grid_pass_{st.session_state.active_user}_{r['id']}"):
                            get_active()["passes"].append(r["id"])
                            log_interaction(st.session_state.active_user, get_active()["settings"]["name"], r, "pass", r.get("compatibility", 0.0))

with tabs[2]:
    st.subheader("Your Decisions")
    ustate = get_active()
    liked_ids = set(ustate["likes"] + ustate["superlikes"])
    liked_df = df_ranked[df_ranked["id"].isin(liked_ids)]
    passed_df = df_ranked[df_ranked["id"].isin(ustate["passes"])]
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
    st.dataframe(df_ranked, use_container_width=True)
    st.info(
        "Replace `compute_compatibility` with your recommender model as needed. "
        "All interactions are appended to the interactions CSV; viewers are persisted to viewers.csv."
    )
