import faiss
import numpy as np
from app.config import FAISS_INDEX_PATH


class VectorDB:

    def __init__(self):

        self.index = faiss.read_index(FAISS_INDEX_PATH)

    def search(self, query_vector, k=3):

        vector = np.array([query_vector]).astype("float32")

        D, I = self.index.search(vector, k)

        return D[0], I[0]