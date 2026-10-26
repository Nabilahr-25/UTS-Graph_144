import pandas as pd

# Periode yang dipilih
PERIODE = {
    "P1": ("2015-05-03", "2015-05-17"),  # training
    "P2": ("2015-07-01", "2015-07-15"),  # testing
}

def ke_ms(tanggal, tambah_hari=0):
    t = pd.Timestamp(tanggal, tz="UTC") + pd.Timedelta(days=tambah_hari)
    return int(t.timestamp() * 1000)

df = pd.read_csv("data/events.csv")
df["transactionid"] = df["transactionid"].astype("Int64")
print("Total baris asli:", len(df))

hasil = []
for nama, (awal, akhir) in PERIODE.items():
    batas_awal = ke_ms(awal)
    batas_akhir = ke_ms(akhir, tambah_hari=1) 
    sub = df[(df["timestamp"] >= batas_awal) & (df["timestamp"] < batas_akhir)].copy()

    sub.to_csv(f"data/events_{nama}.csv", index=False)

    print(f"\n=== {nama}: {awal} s/d {akhir} ===")
    print("Baris (= relasi):", len(sub))
    print("Visitor unik    :", sub["visitorid"].nunique())
    print("Item unik       :", sub["itemid"].nunique())
    print("Total node      :", sub["visitorid"].nunique() + sub["itemid"].nunique())
    print(sub["event"].value_counts().to_string())

    sub["periode"] = nama
    hasil.append(sub)

gabungan = pd.concat(hasil)
gabungan.to_csv("data/events_P1_P2.csv", index=False)
print("\nFile gabungan:", len(gabungan), "baris")