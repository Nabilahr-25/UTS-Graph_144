import argparse
import os
import sys

import pandas as pd
from dotenv import load_dotenv
from neo4j import GraphDatabase
from tqdm import tqdm

load_dotenv()

BATCH_SIZE = 5000


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Impor dataset RetailRocket (events.csv) ke database Neo4j."
    )
    parser.add_argument("-u", "--user", default=os.getenv("NEO4J_USERNAME", "neo4j"),
                        help="Username Neo4j (default: dari .env)")
    parser.add_argument("-p", "--password", default=os.getenv("NEO4J_PASSWORD", "neo4j"),
                        help="Password Neo4j (default: dari .env)")
    parser.add_argument("-b", "--bolt", default=os.getenv("NEO4J_URI", "neo4j://127.0.0.1:7687"),
                        help="Neo4j URI / Bolt (default: dari .env)")
    parser.add_argument("-d", "--database", default=os.getenv("NEO4J_DATABASE", "neo4j"),
                        help="Nama database (default: dari .env)")
    parser.add_argument("-s", "--source", default="data",
                        help="Direktori/folder sumber data (default: data)")
    parser.add_argument("-f", "--file", default="events_P1.csv",
                        help="Nama file CSV (default: events_P1.csv)")
    parser.add_argument("--reset", action="store_true",
                        help="Hapus node Visitor/Item lama sebelum impor")
    return parser.parse_args()


def create_constraints(session):
    session.run(
        "CREATE CONSTRAINT visitor_id IF NOT EXISTS "
        "FOR (v:Visitor) REQUIRE v.visitorId IS UNIQUE"
    )
    session.run(
        "CREATE CONSTRAINT item_id IF NOT EXISTS "
        "FOR (i:Item) REQUIRE i.itemId IS UNIQUE"
    )


def import_batch(tx, batch):
    cypher_query = """
    UNWIND $batch AS row
    MERGE (v:Visitor {visitorId: row.visitorid})
    MERGE (i:Item {itemId: row.itemid})

    WITH v, i, row
    CALL {
        WITH v, i, row
        WITH v, i, row WHERE row.event = 'view'
        CREATE (v)-[:VIEWED {timestamp: row.timestamp}]->(i)

        UNION ALL

        WITH v, i, row
        WITH v, i, row WHERE row.event = 'addtocart'
        CREATE (v)-[:ADDED_TO_CART {timestamp: row.timestamp}]->(i)

        UNION ALL

        WITH v, i, row
        WITH v, i, row WHERE row.event = 'transaction'
        CREATE (v)-[:BOUGHT {timestamp: row.timestamp, transactionId: row.transactionid}]->(i)
    }
    """
    tx.run(cypher_query, batch=batch)


def main():
    args = parse_arguments()

    # 1) Cek file
    csv_file_path = os.path.join(args.source, args.file)
    if not os.path.exists(csv_file_path):
        print(f"Error: File '{csv_file_path}' tidak ditemukan!")
        sys.exit(1)

    # 2) Koneksi
    try:
        driver = GraphDatabase.driver(args.bolt, auth=(args.user, args.password))
        driver.verify_connectivity()
    except Exception as e:
        print(f"Gagal terhubung ke database: {e}")
        sys.exit(1)

    with driver.session(database=args.database) as session:
        # 3) Reset dan penjaga supaya tidak impor dobel
        if args.reset:
            print("Menghapus node Visitor/Item lama...")
            session.run(
                "MATCH (n) WHERE n:Visitor OR n:Item "
                "CALL { WITH n DETACH DELETE n } IN TRANSACTIONS OF 10000 ROWS"
            )
        ada = session.run("MATCH (v:Visitor) RETURN count(v) AS n").single()["n"]
        if ada > 0:
            print(f"Sudah ada {ada} node Visitor. Jalankan ulang dengan --reset kalau mau impor ulang.")
            driver.close()
            sys.exit(1)

        create_constraints(session)

        # 4) Impor batch
        print(f"Membaca file data dari: {csv_file_path}")
        total_rows = sum(1 for _ in open(csv_file_path, encoding="utf-8")) - 1
        total_chunks = (total_rows // BATCH_SIZE) + 1
        chunks = pd.read_csv(csv_file_path, chunksize=BATCH_SIZE,
                             dtype={"transactionid": "string"})

        print(f"Memulai proses impor {total_rows} baris ke Neo4j...")
        for chunk in tqdm(chunks, total=total_chunks, desc="Mengimpor Batch Data"):
            # Membersihkan data NaN pada transactionid
            chunk["transactionid"] = chunk["transactionid"].fillna("").astype(str)
            records = chunk.to_dict(orient="records")
            session.execute_write(import_batch, records)

        # 5) Ringkasan hasil
        print("\nNode per label:")
        for rec in session.run(
            "MATCH (n) UNWIND labels(n) AS label RETURN label, count(*) AS jumlah ORDER BY jumlah DESC"
        ):
            print(" ", rec["label"], rec["jumlah"])
        print("Relasi per tipe:")
        for rec in session.run(
            "MATCH ()-[r]->() RETURN type(r) AS tipe, count(*) AS jumlah ORDER BY jumlah DESC"
        ):
            print(" ", rec["tipe"], rec["jumlah"])

    driver.close()
    print("Proses impor berhasil.")


if __name__ == "__main__":
    main()