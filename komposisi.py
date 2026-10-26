import os

import pandas as pd
from dotenv import load_dotenv
from neo4j import GraphDatabase

load_dotenv()
DB = os.getenv("NEO4J_DATABASE", "neo4j")

# 1) Jenis simpul dan jumlahnya
QUERY_NODE = """
MATCH (n)
UNWIND labels(n) AS label
RETURN label, count(*) AS jumlah
ORDER BY jumlah DESC
"""

# 2) Jenis relasi, jumlahnya, dan berapa visitor/item unik yang terlibat
QUERY_REL = """
MATCH (v:Visitor)-[r]->(i:Item)
RETURN type(r) AS tipe_relasi,
       count(r) AS jumlah_relasi,
       count(DISTINCT v) AS visitor_unik,
       count(DISTINCT i) AS item_unik
ORDER BY jumlah_relasi DESC
"""

# 3) Total keseluruhan
QUERY_TOTAL = """
MATCH (n)
WITH count(n) AS total_node
MATCH ()-[r]->()
RETURN total_node, count(r) AS total_relasi
"""


def jalankan(driver, query):
    records, _, _ = driver.execute_query(query, database_=DB)
    return pd.DataFrame([r.data() for r in records])


with GraphDatabase.driver(
    os.getenv("NEO4J_URI"),
    auth=(os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD")),
) as driver:
    print("=== Jumlah simpul per jenis ===")
    print(jalankan(driver, QUERY_NODE).to_string(index=False))

    print("\n=== Relasi (arah Visitor -> Item) ===")
    print(jalankan(driver, QUERY_REL).to_string(index=False))

    print("\n=== Total ===")
    print(jalankan(driver, QUERY_TOTAL).to_string(index=False))