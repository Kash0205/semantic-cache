from app.utils.similarity import cosine_similarity
from app.config import SIMILARITY_THRESHOLD


class SemanticCache:

    def __init__(self):

        self.cache = []

        self.hit_count = 0
        self.miss_count = 0

    def lookup(self, embedding, cluster):

        best_sim = 0
        best_entry = None

        for entry in self.cache:

            if entry["cluster"] != cluster:
                continue

            sim = cosine_similarity(embedding, entry["embedding"])

            if sim > best_sim:
                best_sim = sim
                best_entry = entry

        if best_sim >= SIMILARITY_THRESHOLD:

            self.hit_count += 1

            return best_entry, best_sim

        self.miss_count += 1

        return None, None

    def add(self, query, embedding, result, cluster):

        self.cache.append(
            {
                "query": query,
                "embedding": embedding,
                "result": result,
                "cluster": cluster
            }
        )

    def stats(self):

        total = self.hit_count + self.miss_count

        return {
            "total_entries": len(self.cache),
            "hit_count": self.hit_count,
            "miss_count": self.miss_count,
            "hit_rate": self.hit_count / total if total else 0
        }

    def clear(self):

        self.cache = []
        self.hit_count = 0
        self.miss_count = 0