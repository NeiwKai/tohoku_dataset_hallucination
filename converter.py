import pandas as pd

# Read CSV
df = pd.read_csv("conference_paper.csv")

# Split into title and year
df[["title", "year"]] = df["conference_source"].str.rsplit(
    ",", n=1, expand=True
)

# Remove leading/trailing whitespace
df["title"] = df["title"].str.strip()
df["year"] = df["year"].str.strip()

df = df.drop(columns=["conference_source"])

# Save result
df.to_csv("conference_paper.csv", index=False, quoting=1)
