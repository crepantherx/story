
# pages/Explore Profiles.py — an integrated "all profiles" explorer page
import os, ast, json
import pandas as pd
import streamlit as st

st.set_page_config(layout="wide", page_title="Explore Profiles")

st.title("🌍 Explore Profiles")
st.caption("This page reads from the main app's dataset (session_state).")

def parse_list_cols(df: pd.DataFrame, cols=("languages","interests")) -> pd.DataFrame:
    for col in cols:
        if col in df.columns and df[col].dtype == object:
            def _parse(v):
                if isinstance(v, list):
                    return v
                try:
                    x = ast.literal_eval(v)
                    return x if isinstance(x, list) else [str(x)]
                except Exception:
                    return [str(v)] if pd.notna(v) else []
            df[col] = df[col].apply(_parse)
    return df

def get_data() -> pd.DataFrame:
    # Preferred: use the dataset already created by the main app
    if "profiles_df" in st.session_state:
        return parse_list_cols(st.session_state.profiles_df.copy())
    # Fallback 1: environment CSV path
    csv_path = os.environ.get("WORLD_PROFILES_CSV", "world_profiles.csv")
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        return parse_list_cols(df)
    # Fallback 2: minimal in-app generation to avoid breaking the page
    st.warning("No in-memory dataset found and CSV missing. Generating a small sample (200 rows).", icon="⚠️")
    try:
        # Try to import the generator if app.py placed it in sys.modules
        from app import make_world_profiles  # type: ignore
        return make_world_profiles(n=200, seed=123)
    except Exception:
        # Ultra-minimal random sample if import fails
        import random, uuid
        rng = random.Random(123)
        rows = []
        for _ in range(200):
            pid = str(uuid.uuid4())[:8]
            rows.append({
                "id": pid, "name": rng.choice(["Aditi","Rahul","Sam","Neha","Rohan","Diya"]),
                "age": rng.randint(21,45),
                "gender": rng.choice(["Woman","Man","Non-binary"]),
                "region": rng.choice(["South Asia","Europe","North America"]),
                "country": rng.choice(["India","UK","USA"]),
                "city": rng.choice(["Mumbai","Delhi","Bengaluru","London","New York"]),
                "distance_km": rng.randint(1,30),
                "interests": random.sample(["Music","Travel","Foodie","Tech","Yoga","Hiking","Movies"], k=3),
                "about":"Quick placeholder profile.",
                "photo_url": f"https://picsum.photos/seed/{pid}/480/480",
            })
        return pd.DataFrame(rows)

df = get_data()

if df.empty:
    st.info("No data to show yet. Open the main app page first to generate/load a dataset.", icon="ℹ️")
    st.stop()

# ---------------------------
# Sidebar filters
# ---------------------------
with st.sidebar:
    st.header("🔎 Filters")
    regions = sorted(df["region"].dropna().unique().tolist()) if "region" in df.columns else []
    sel_regions = st.multiselect("Region", regions, default=regions)

    df_r = df[df["region"].isin(sel_regions)] if sel_regions and "region" in df.columns else df.copy()

    countries = sorted(df_r["country"].dropna().unique().tolist()) if "country" in df_r.columns else []
    sel_countries = st.multiselect("Country", countries)

    df_c = df_r[df_r["country"].isin(sel_countries)] if sel_countries and "country" in df_r.columns else df_r

    cities = sorted(df_c["city"].dropna().unique().tolist()) if "city" in df_c.columns else []
    sel_cities = st.multiselect("City", cities)

    genders = sorted(df["gender"].dropna().unique().tolist()) if "gender" in df.columns else []
    sel_genders = st.multiselect("Gender", genders, default=genders)

    # Numeric sliders (guard if columns exist)
    age_min = int(df["age"].min()) if "age" in df.columns else 18
    age_max = int(df["age"].max()) if "age" in df.columns else 80
    s_age = st.slider("Age range", min_value=age_min, max_value=age_max, value=(age_min, age_max))

    dist_min = int(df["distance_km"].min()) if "distance_km" in df.columns else 0
    dist_max = int(df["distance_km"].max()) if "distance_km" in df.columns else 30
    s_dist = st.slider("Distance (km)", min_value=dist_min, max_value=dist_max, value=(dist_min, dist_max))

    # Interests
    if "interests" in df.columns:
        all_interests = sorted({i for L in df["interests"].dropna().tolist() for i in (L if isinstance(L, list) else [])})
    else:
        all_interests = []
    sel_interests = st.multiselect("Interests (match ANY)", all_interests)

    q = st.text_input("Search name/about", placeholder="e.g., hiking, Mumbai, music")

# Apply filters safely
mask = pd.Series([True] * len(df))
if sel_regions and "region" in df.columns:
    mask &= df["region"].isin(sel_regions)
if sel_countries and "country" in df.columns:
    mask &= df["country"].isin(sel_countries)
if sel_cities and "city" in df.columns:
    mask &= df["city"].isin(sel_cities)
if sel_genders and "gender" in df.columns:
    mask &= df["gender"].isin(sel_genders)
if "age" in df.columns:
    mask &= df["age"].between(*s_age)
if "distance_km" in df.columns:
    mask &= df["distance_km"].between(*s_dist)
if sel_interests and "interests" in df.columns:
    mask &= df["interests"].apply(lambda L: bool(set(L) & set(sel_interests)))
if q:
    ql = q.lower().strip()
    name_m = df["name"].str.lower().str.contains(ql, na=False) if "name" in df.columns else False
    about_m = df["about"].str.lower().str.contains(ql, na=False) if "about" in df.columns else False
    city_m = df["city"].str.lower().str.contains(ql, na=False) if "city" in df.columns else False
    ctry_m = df["country"].str.lower().str.contains(ql, na=False) if "country" in df.columns else False
    mask &= (name_m | about_m | city_m | ctry_m)

filtered = df[mask].copy()

# KPIs
k1, k2, k3, k4 = st.columns(4)
k1.metric("Profiles", len(filtered))
k2.metric("Countries", filtered["country"].nunique() if "country" in filtered.columns else 0)
k3.metric("Cities", filtered["city"].nunique() if "city" in filtered.columns else 0)
k4.metric("Avg Age", f"{filtered['age'].mean():.1f}" if "age" in filtered.columns and not filtered.empty else "—")

st.divider()

# Charts
c1, c2 = st.columns(2)
with c1:
    st.subheader("Age distribution")
    if "age" in filtered.columns and not filtered.empty:
        hist = filtered["age"].value_counts().sort_index()
        st.bar_chart(hist, use_container_width=True)
    else:
        st.info("No data for chart.")

with c2:
    st.subheader("Top interests")
    if "interests" in filtered.columns and not filtered.empty:
        exploded = filtered.explode("interests")
        topI = exploded["interests"].value_counts().head(15)
        st.bar_chart(topI, use_container_width=True)
    else:
        st.info("No data for chart.")

st.divider()

# Gallery
st.subheader("Profiles")
if filtered.empty:
    st.info("No matching profiles.")
else:
    cols = st.columns(4)
    for i, (_, row) in enumerate(filtered.head(48).iterrows()):
        with cols[i % 4]:
            st.image(row.get("photo_url", ""), caption=row.get("name",""), use_container_width=True)
            age = int(row["age"]) if "age" in row else "—"
            city = row.get("city","")
            country = row.get("country","")
            st.write(f"**{row.get('name','')}**, {age} • {row.get('gender','')}")
            st.caption(f"📍 {city}{', ' if city and country else ''}{country}")
            ints = row.get("interests", [])
            if isinstance(ints, list):
                st.write(", ".join(ints[:3]))

# Data + downloads
st.subheader("Data")
existing_cols = [c for c in ["id","name","age","gender","region","country","city","distance_km","about","interests","photo_url"] if c in filtered.columns]
st.dataframe(filtered[existing_cols], use_container_width=True)

def to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")

def to_jsonl_bytes(df: pd.DataFrame) -> bytes:
    lines = []
    for _, r in df.iterrows():
        obj = r.to_dict()
        lines.append(json.dumps(obj, ensure_ascii=False))
    return ("\n".join(lines)).encode("utf-8")

c1, c2 = st.columns(2)
with c1:
    st.download_button("⬇️ Download filtered as CSV", data=to_csv_bytes(filtered), file_name="profiles_filtered.csv", mime="text/csv")
with c2:
    st.download_button("⬇️ Download filtered as JSONL", data=to_jsonl_bytes(filtered), file_name="profiles_filtered.jsonl", mime="application/json")
