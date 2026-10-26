import os

import numpy as np
import pandas as pd
import scipy.sparse as sp
from dotenv import load_dotenv
from neo4j import GraphDatabase
from tqdm import tqdm

load_dotenv()
DB = os.getenv("NEO4J_DATABASE", "neo4j")
K = 5
BATCH = 5000

# 1) Muat matriks user x item hasil vsm.py, lalu balik jadi item x user
M = sp.load_npz("data/M_user_item.npz")
item_ids = np.load("data/item_ids.npy")
X = M.T.tocsr()

# 2) Normalisasi L2 tiap vektor item, lalu cosine = perkalian titik
norm = np.sqrt(X.multiply(X).sum(axis=1)).A1
X = (sp.diags(1.0 / norm) @ X).tocsr()
S = (X @ X.T).tocsr()
S.setdiag(0)
S.eliminate_zeros()
print("Pasangan item dengan similarity > 0 (terarah):", S.nnz)

# 3) Ambil top-K tetangga tiap item (seri diurutkan berdasarkan itemId supaya reproducible)
rows = []
for r in range(S.shape[0]):
    a, b = S.indptr[r], S.indptr[r + 1]
    if a == b:
        continue
    idx = S.indices[a:b]
    sc = S.data[a:b]
    urut = np.lexsort((item_ids[idx], -sc))[:K]
    for rank, o in enumerate(urut, start=1):
        rows.append((int(item_ids[r]), int(item_ids[idx[o]]), round(float(sc[o]), 6), rank))

df = pd.DataFrame(rows, columns=["item_a", "item_b", "score", "rank"])
df.to_csv("data/similarity_top5.csv", index=False)
print("Item yang punya tetangga:", df["item_a"].nunique())
print("Pasangan top-5 disimpan:", len(df))

# 4) Tulis balik ke Neo4j sebagai relasi SIMILAR_TO
QUERY = """
UNWIND $rows AS row
MATCH (a:Item {itemId: row.a})
MATCH (b:Item {itemId: row.b})
MERGE (a)-[s:SIMILAR_TO]->(b)
SET s.score = row.score, s.rank = row.rank
"""
data = [
    {"a": int(a), "b": int(b), "score": float(s), "rank": int(k)}
    for a, b, s, k in df.itertuples(index=False)
]

with GraphDatabase.driver(
    os.getenv("NEO4J_URI"),
    auth=(os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD")),
) as driver:
    with driver.session(database=DB) as session:
        for i in tqdm(range(0, len(data), BATCH), desc="Menulis SIMILAR_TO"):
            session.run(QUERY, rows=data[i:i + BATCH]).consume()

        total = session.run("MATCH ()-[s:SIMILAR_TO]->() RETURN count(s) AS n").single()["n"]
        print("Relasi SIMILAR_TO di Neo4j:", total)