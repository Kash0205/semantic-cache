
import faiss
import numpy as np

embeddings = np.load("storage/embeddings.npy")

dimension = embeddings.shape[1]

index = faiss.IndexFlatIP(dimension)

faiss.normalize_L2(embeddings)

index.add(embeddings)

faiss.write_index(index, "storage/faiss.index")

print("FAISS index built")