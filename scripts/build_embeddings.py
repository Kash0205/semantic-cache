import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import pickle
import numpy as np
from app.embeddings.embedder import embed_batch

with open("storage/documents.pkl", "rb") as f:
    docs = pickle.load(f)

embeddings = embed_batch(docs)

np.save("storage/embeddings.npy", embeddings)

print("Embeddings saved")