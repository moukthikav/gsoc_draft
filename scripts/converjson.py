import pandas as pd

df = pd.read_csv("data/hindi_triples_500.csv")

# Use force_ascii=False to keep the Hindi characters visible
df.to_json("data/training_data.json", orient="records", indent=4, force_ascii=False)

print("Done! Open the file now, it will be in Hindi.")