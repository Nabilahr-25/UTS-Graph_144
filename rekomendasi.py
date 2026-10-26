import os

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from neo4j import GraphDatabase

load_dotenv()
DB = os.getenv("NEO4J_DATABASE", "neo4j")
SEED = 42
BOBOT = {"view": 1, "addtocart": 3, "transaction": 5}

p1 = pd.read_csv("data/events_P1.csv")
p2 = pd.read_csv("data/events_P2.csv")
sim = pd.read_csv("data/similarity_top5.csv")

# Item per visitor di P1 (untuk riwayat) dan di P2 (untuk kunci jawaban)
p1["bobot"] = p1["event"].map(BOBOT)
bobot_p1 = p1.groupby(["visitorid", "itemid"], as_index=False)["bobot"].sum()
items_p1 = bobot_p1.groupby("visitorid")["itemid"].apply(set).to_dict()
items_p2 = p2.groupby("visitorid")["itemid"].apply(set).to_dict()
tetangga = sim.groupby("item_a")["item_b"].apply(set).to_dict()

# Kumpulan user kandidat
kandidat = []
for uid in sorted(set(items_p1) & set(items_p2)):
    sudah = items_p1[uid]
    kunci = items_p2[uid] - sudah
    calon = set().union(*[tetangga.get(i, set()) for i in sudah]) - sudah
    if kunci and calon:
        kandidat.append(int(uid))
print("User kandidat (ada di P1 dan P2, bisa dievaluasi):", len(kandidat))

# Pilih user acak, reproducible
rng = np.random.default_rng(SEED)
user = int(rng.choice(kandidat))
print("User terpilih (seed 42):", user)

print("\nRiwayat user di P1:")
print(bobot_p1[bobot_p1["visitorid"] == user][["itemid", "bobot"]].to_string(index=False))

# KNN item-based lewat Cypher
QUERY = """
MATCH (v:Visitor {visitorId: $uid})-[r]->(i:Item)
WITH v, i, sum(CASE type(r) WHEN 'VIEWED' THEN 1 WHEN 'ADDED_TO_CART' THEN 3 WHEN 'BOUGHT' THEN 5 END) AS bobot
MATCH (i)-[s:SIMILAR_TO]->(j:Item)
WHERE NOT (v)-->(j)
RETURN j.itemId AS item, sum(bobot * s.score) AS skor, count(i) AS pendukung
ORDER BY skor DESC, item
LIMIT 5
"""

with GraphDatabase.driver(
    os.getenv("NEO4J_URI"),
    auth=(os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD")),
) as driver:
    records, _, _ = driver.execute_query(QUERY, uid=user, database_=DB)

rek = pd.DataFrame([r.data() for r in records])
rek["peringkat"] = range(1, len(rek) + 1)
rek["skor"] = rek["skor"].round(6)
rek.to_csv("data/rekomendasi_user.csv", index=False)

print("\nTop-5 rekomendasi (k = 5):")
print(rek.to_string(index=False))
print("\n3 item yang diiklankan:", rek["item"].head(3).tolist())