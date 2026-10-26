from collections import defaultdict

import numpy as np
import pandas as pd

USER_ID = 121532  # hasil pemilihan acak seed 42 dari rekomendasi.py
BOBOT = {"view": 1, "addtocart": 3, "transaction": 5}

p1 = pd.read_csv("data/events_P1.csv")
p2 = pd.read_csv("data/events_P2.csv")
sim = pd.read_csv("data/similarity_top5.csv")

p1["bobot"] = p1["event"].map(BOBOT)
bobot_p1 = p1.groupby(["visitorid", "itemid"], as_index=False)["bobot"].sum()
riwayat = {int(u): dict(zip(g["itemid"], g["bobot"])) for u, g in bobot_p1.groupby("visitorid")}
items_p2 = {int(u): set(s) for u, s in p2.groupby("visitorid")["itemid"]}

tetangga = defaultdict(list)
for a, b, s, _ in sim.itertuples(index=False):
    tetangga[int(a)].append((int(b), float(s)))


def rekomendasi(uid, k=5):
    sudah = riwayat[uid]
    skor = defaultdict(float)
    for item, bobot in sudah.items():
        for j, s in tetangga.get(int(item), []):
            if j not in sudah:
                skor[j] += bobot * s
    urut = sorted(skor.items(), key=lambda x: (-x[1], x[0]))
    return [j for j, _ in urut[:k]]


def hit_at_k(top, kunci, k):
    return int(any(x in kunci for x in top[:k]))


def ndcg_at_k(top, kunci, k):
    dcg = sum(1 / np.log2(p + 2) for p, x in enumerate(top[:k]) if x in kunci)
    idcg = sum(1 / np.log2(p + 2) for p in range(min(len(kunci), k)))
    return dcg / idcg if idcg > 0 else 0.0


# A) User terpilih
kunci = items_p2[USER_ID] - set(riwayat[USER_ID])
top = pd.read_csv("data/rekomendasi_user.csv")["item"].tolist()  # hasil Cypher dari Neo4j
print("=== User terpilih:", USER_ID, "===")
print("Kunci jawaban (item baru di P2):", sorted(kunci))
print("Rekomendasi top-5               :", top)
print("Hit@5  :", hit_at_k(top, kunci, 5))
print("NDCG@5 :", round(ndcg_at_k(top, kunci, 5), 4))
print("Hit@3 (3 item yang diiklankan):", hit_at_k(top, kunci, 3))

# B) Semua user kandidat (pembanding)
populer = (
    bobot_p1.groupby("itemid", as_index=False)["bobot"].sum()
    .sort_values(["bobot", "itemid"], ascending=[False, True])["itemid"].astype(int).tolist()
)
h5, n5, h3, ph5, pn5 = [], [], [], [], []
for uid in sorted(set(riwayat) & set(items_p2)):
    kunci_u = items_p2[uid] - set(riwayat[uid])
    rek = rekomendasi(uid)
    if not kunci_u or not rek:
        continue
    h5.append(hit_at_k(rek, kunci_u, 5))
    n5.append(ndcg_at_k(rek, kunci_u, 5))
    h3.append(hit_at_k(rek, kunci_u, 3))
    pop = [x for x in populer if x not in riwayat[uid]][:5]
    ph5.append(hit_at_k(pop, kunci_u, 5))
    pn5.append(ndcg_at_k(pop, kunci_u, 5))

print("\n=== Semua user kandidat:", len(h5), "===")
print(f"Item-based KNN : Hit@5 {np.mean(h5):.4f} ({sum(h5)} user), NDCG@5 {np.mean(n5):.4f}, Hit@3 {np.mean(h3):.4f}")
print(f"Baseline populer: Hit@5 {np.mean(ph5):.4f} ({sum(ph5)} user), NDCG@5 {np.mean(pn5):.4f}")