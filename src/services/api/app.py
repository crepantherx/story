from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def home():
    return {"status": "ok"}


@app.get("/match/{profile}")
def match(profile: str):
    def a(profile):
        from pinecone import Pinecone

        pc = Pinecone(api_key="pcsk_61UNS7_CWf4kKYfMMbpSf3HgMmtaqMtXZYwemNJRR7b7RUcKc5RioQgNbCdmWd5sCgRx73")

        index = pc.Index("profiles")

        response = index.fetch(ids=[profile])

        # supabase_insert_matches.py
        import os
        import sys
        from supabase import create_client, Client

        # --- config / client ---
        url: str = "https://tpquhacpoxoschgsarie.supabase.co"
        key: str = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InRwcXVoYWNwb3hvc2NoZ3NhcmllIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NjMyMTk3NjksImV4cCI6MjA3ODc5NTc2OX0.T06IB1qnCr8eL1BCvuSypVkS7Cgeu5wdnE8QrSWmb-w"

        if not url or not key:
            print("Please set SUPABASE_URL and SUPABASE_KEY environment variables.", file=sys.stderr)
            sys.exit(1)

        supabase: Client = create_client(url, key)

        import pandas as pd

        l = (supabase.table("interactions")
            .select("*")
            .filter("viewer_id", "ilike", f"%{profile}%")
            .execute()
            .data)

        if l:
            already_seen = pd.DataFrame().loc[:, 'profile_id'].to_list()
        else:
            already_seen = l

        result = index.query(
            vector=response.vectors[profile].values,
            top_k=1,
            include_metadata=True,
            filter={
                "profile_id": {"$nin": already_seen}
            }
        )

        top1 = result.matches[0]
        return top1.id

    return {"best_match": a(profile)}

