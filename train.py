import polars as pl

df = pl.read_json("./data_snapshot_formatted.json")

print(len(df["Class_names"].value_counts()["Class_names"]))
