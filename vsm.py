import os

import matplotlib
matplotlib.use("Agg")  # simpan gambar ke file, tanpa membuka jendela
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import scipy.sparse as sp
from dotenv import load_dotenv
from neo4j import GraphDatabase
from scipy.sparse.csgraph import connected_components

load_dotenv()
DB = os.getenv("NEO4J_DATABASE", "neo4j")
BOBOT = {"VIEWED": 1, "ADDED_TO_CART": 3, "BOUGHT": 5}
os.makedirs("hasil", exist_ok=True)

QUERY = """
MATCH (v:Visitor)-[r]->(i:Item)
RETURN v.visitorId AS visitorid, i.itemId AS itemid, type(r) AS tipe, count(r) AS n
"""

# 1) Ambil relasi dari Neo4j
with GraphDatabase.driver(
    os.getenv("NEO4J_URI"),
    auth=(os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD")),
) as driver:
    records, _, _ = driver.execute_query(QUERY, database_=DB)

edges = pd.DataFrame([r.data() for r in records])
edges["skor"] = edges["n"] * edges["tipe"].map(BOBOT)
agg = edges.groupby(["visitorid", "itemid"], as_index=False)["skor"].sum()
print("Pasangan (visitor, item) unik:", len(agg))

# 2) Matriks user x item 
user_ids, ui = np.unique(agg["visitorid"], return_inverse=True)
item_ids, ii = np.unique(agg["itemid"], return_inverse=True)
M = sp.csr_matrix(
    (agg["skor"].to_numpy(dtype=np.float32), (ui, ii)),
    shape=(len(user_ids), len(item_ids)),
)

sp.save_npz("data/M_user_item.npz", M)
np.save("data/user_ids.npy", user_ids)
np.save("data/item_ids.npy", item_ids)


def karakter(nama, X, label_dim):
    n, d = X.shape
    per_vektor = np.diff(X.indptr)
    print(f"\n=== VSM {nama} ===")
    print(f"Shape            : {X.shape}")
    print(f"Dimensi vektor   : {d} ({label_dim})")
    print(f"Non-zero         : {X.nnz}")
    print(f"Sparsity         : {100 * (1 - X.nnz / (n * d)):.4f}%")
    print(f"Non-zero/vektor  : mean {per_vektor.mean():.2f}, median {np.median(per_vektor):.0f}, max {per_vektor.max()}")
    print(f"Vektor 1 non-zero: {100 * (per_vektor == 1).mean():.1f}%")
    print(f"Memori sparse    : {(X.data.nbytes + X.indices.nbytes + X.indptr.nbytes) / 1e6:.1f} MB")
    print(f"Memori dense     : {n * d * 4 / 1e9:.1f} GB (float32)")


def proyeksi(nama, X):
    B = X.copy()
    B.data[:] = 1                      
    A = (B @ B.T).tocsr()             
    A.setdiag(0)
    A.eliminate_zeros()
    n = A.shape[0]
    deg = np.diff(A.indptr)
    n_komp = connected_components(A, directed=False)[0]
    print(f"\n--- Proyeksi {nama} ---")
    print(f"Node             : {n}")
    print(f"Edge             : {A.nnz // 2}")
    print(f"Densitas         : {A.nnz / (n * n):.6f}")
    print(f"Node terisolasi  : {(deg == 0).sum()}")
    print(f"Komponen terhubung: {n_komp}")
    print(f"Derajat max      : {deg.max()}")
    return A


def plot_ego(A, ids, judul, nama_file, maks=30):
    deg = np.diff(A.indptr)
    ego = int(np.argmax(deg))
    row = A.getrow(ego)
    top = row.indices[np.argsort(-row.data)[:maks]]
    nodes = [ego] + list(top)
    sub = A[nodes][:, nodes]
    G = nx.from_scipy_sparse_array(sub)
    G = nx.relabel_nodes(G, {k: str(ids[v]) for k, v in enumerate(nodes)})
    warna = ["red" if n == str(ids[ego]) else "lightblue" for n in G.nodes]
    pos = nx.spring_layout(G, seed=42)
    plt.figure(figsize=(9, 7))
    nx.draw_networkx(G, pos, node_color=warna, node_size=250, font_size=6, width=0.5)
    plt.title(f"{judul} (merah = node dengan derajat tertinggi, {len(nodes)} node)")
    plt.axis("off")
    plt.savefig(nama_file, dpi=150, bbox_inches="tight")
    plt.close()


# 3) VSM User
karakter("User (user x item)", M, "item")
A_user = proyeksi("User-User", M)
plot_ego(A_user, user_ids, "Proyeksi User-User", "hasil/plot_user.png")

# 4) VSM Item
M_item = M.T.tocsr()
karakter("Item (item x user)", M_item, "user")
A_item = proyeksi("Item-Item", M_item)
plot_ego(A_item, item_ids, "Proyeksi Item-Item", "hasil/plot_item.png")

print("\nSelesai. Gambar ada di folder hasil/")