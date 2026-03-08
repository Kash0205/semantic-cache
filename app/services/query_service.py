import pickle
from app.embeddings.embedder import embed_text
from app.vector_db.faiss_index import VectorDB
from app.clustering.fuzzy_cluster import FuzzyCluster
from app.cache.semantic_cache import SemanticCache
from app.config import DATASET_PATH, TOP_K_RESULTS

vector_db = VectorDB()
cluster_model = FuzzyCluster()
cache = SemanticCache()

with open(DATASET_PATH, "rb") as f:
    documents = pickle.load(f)


def handle_query(query):

    query_embedding = embed_text(query)

    cluster = cluster_model.predict_cluster(query_embedding)

    entry, sim = cache.lookup(query_embedding, cluster)

    if entry:

        return {
            "query": query,
            "cache_hit": True,
            "matched_query": entry["query"],
            "similarity_score": float(sim),
            "result": entry["result"],
            "dominant_cluster": cluster
        }

    # VECTOR SEARCH

    D, I = vector_db.search(query_embedding, TOP_K_RESULTS)

    result = [documents[i] for i in I]

    cache.add(query, query_embedding, result, cluster)

    return {
        "query": query,
        "cache_hit": False,
        "matched_query": None,
        "similarity_score": None,
        "result": result,
        "dominant_cluster": cluster
    }