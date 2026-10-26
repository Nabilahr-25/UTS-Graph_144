import pandas as pd

df = pd.read_csv("data/events.csv")

print("Shape:", df.shape)
print(df.dtypes)
print(df.head())

print("\nJumlah per event:")
print(df["event"].value_counts())

df["waktu"] = pd.to_datetime(df["timestamp"], unit="ms")
print("\nAwal :", df["waktu"].min())
print("Akhir:", df["waktu"].max())

print("\nVisitor unik:", df["visitorid"].nunique())
print("Item unik   :", df["itemid"].nunique())

print("\nEvent per bulan:")
print(df.groupby(df["waktu"].dt.to_period("M")).size())