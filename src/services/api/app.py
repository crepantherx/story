from fastapi import FastAPI, Depends, HTTPException, status, Request
app = FastAPI()
from annoy import AnnoyIndex
import pandas as pd
import numpy as np

def build_index():
    import re
    import pandas as pd
    import numpy as np

    def str_to_array(s):
        if s is None:
            return None
        s = s.strip()
        if s.startswith('[') and s.endswith(']'):
            s = s[1:-1]
        s = re.sub(r'[\r\n]+', ' ', s)
        s = re.sub(r'\s+', ' ', s).strip()
        return np.fromstring(s, sep=' ').astype(float)

    df = pd.read_csv("/Users/sudhirsingh/PyCharmProjects/story/src/services/frontend/data/profile_embedding.csv")
    df['embedding'] = df['embedding'].apply(str_to_array)
    vector_dim = len(df.iloc[0]['embedding'])

    annoy_index = AnnoyIndex(vector_dim, 'euclidean')

    id_map = {}

    for idx, row in enumerate(df.itertuples()):
        annoy_index.add_item(idx, row.embedding)
        id_map[idx] = row.id

    annoy_index.build(10)
    return annoy_index, id_map

annoy_index, id_map = build_index()

def authenticate(request: Request):

    password = request.headers.get("Authorization")

    password = password.split(" ")[-1]

    if password != "sudhir@123":
        raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Not Authenticated",
                headers={"WWW-Authenticate": "Basic"}
            )

@app.get("/")
def home():
    return {"status": "ok"}

@app.get("/match/{profile}")
def abc(profile: str):

    def top_n_similar(annoy_index, id_map, query_id, n=5, seen_ids=None):
        query_idx = [k for k, v in id_map.items() if v == query_id][0]
        import re
        def str_to_array(s):
            if s is None:
                return None
            s = s.strip()
            if s.startswith('[') and s.endswith(']'):
                s = s[1:-1]
            s = re.sub(r'[\r\n]+', ' ', s)
            s = re.sub(r'\s+', ' ', s).strip()
            return np.fromstring(s, sep=' ').astype(float)

        df = pd.read_csv("/Users/sudhirsingh/PyCharmProjects/story/src/services/frontend/data/profile_embedding.csv")
        df['embedding'] = df['embedding'].apply(str_to_array)
        embedding = df.loc[df['id'] == query_id, 'embedding'].values[0]
        candidates = annoy_index.get_nns_by_vector(embedding, n + 10)
        result_ids = []
        for i in candidates:
            candidate_id = id_map[i]
            if candidate_id != query_id and (seen_ids is None or candidate_id not in seen_ids):
                result_ids.append(candidate_id)
            if len(result_ids) >= n:
                break
        return result_ids

    def already_seen(query_id):
        interactions = pd.read_csv("/Users/sudhirsingh/PyCharmProjects/story/src/services/frontend/data/interactions.csv",
                                   names=['datetime', 'viewer_id', 'viewer_name', 'profile_id', 'profile_name',
                                          'status', 'score'])
        update = interactions.groupby('viewer_id')['profile_id'].apply(list).reset_index()
        update['viewer_id'] = update['viewer_id'].apply(lambda w: w.split("-")[-1])
        matched_profiles = update.loc[update['viewer_id'] == query_id, 'profile_id'].to_list()
        if matched_profiles:
            seen_ids = matched_profiles[0]
        else:
            seen_ids = None
        return seen_ids

    query_id = profile
    seen_ids = already_seen(query_id)
    results = top_n_similar(annoy_index, id_map, query_id, n=5, seen_ids=seen_ids)

    return {"profile": results[0]}

@app.get("/update/{profile}")
def update(profile: str):
    query_id = profile
    interactions = pd.read_csv(
        "/Users/sudhirsingh/PyCharmProjects/story/src/services/frontend/data/interactions.csv",
        names=['datetime', 'viewer_id', 'viewer_name', 'profile_id', 'profile_name', 'status', 'score'])
    interactions['viewer_id'] = interactions['viewer_id'].apply(lambda w: w.split("-")[-1])
    update = interactions.groupby('viewer_id')['profile_id'].apply(list).reset_index()

    need_to_update = update.loc[update['viewer_id'] == query_id, 'profile_id'].to_list()
    if need_to_update:
        import re
        import numpy as np
        from numpy.linalg import norm

        def str_to_array(s):
            if s is None:
                return None
            s = s.strip()
            if s.startswith('[') and s.endswith(']'):
                s = s[1:-1]
            s = re.sub(r'[\r\n]+', ' ', s)
            s = re.sub(r'\s+', ' ', s).strip()
            return np.fromstring(s, sep=' ').astype(float)

        def avg(l):

            es = p[p['id'].isin(l)]['embedding'].values
            avg = np.mean(es, axis=0)
            n = avg / norm(avg)
            return n

        p = pd.read_csv("/Users/sudhirsingh/PyCharmProjects/story/src/services/frontend/data/profile_embedding.csv")
        p['embedding'] = p['embedding'].apply(str_to_array)

        update['updated_embedding'] = update['profile_id'].apply(avg)

        update = update.rename(columns={'viewer_id': 'id'})
        update = update.rename(columns={'updated_embedding': 'embedding'})
        update = update[['id', 'embedding']]
        update_dict = dict(zip(update['id'], update['embedding']))
        p['embedding'] = p.apply(
            lambda row: update_dict.get(row['id'], row['embedding']),
            axis=1
        )
        p.to_csv("/Users/sudhirsingh/PyCharmProjects/story/src/services/frontend/data/profile_embedding.csv",
                 index=False)
        return {"trained": "ok"}
    else:
        return {"trained": "no"}