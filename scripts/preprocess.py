import pickle
from sklearn.datasets import fetch_20newsgroups
import os

os.makedirs("storage", exist_ok=True)

data = fetch_20newsgroups(
    subset="all",
    remove=("headers", "footers", "quotes")
)

documents = data.data

print("Total documents:", len(documents))

with open("storage/documents.pkl", "wb") as f:
    pickle.dump(documents, f)

print("Dataset saved successfully")