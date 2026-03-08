import joblib
from app.config import CLUSTER_MODEL_PATH


class FuzzyCluster:

    def __init__(self):

        self.model = joblib.load(CLUSTER_MODEL_PATH)

    def predict_cluster(self, embedding):

        cluster = self.model.predict([embedding])[0]

        return int(cluster)

    def cluster_distribution(self, embedding):

        probs = self.model.predict_proba([embedding])[0]

        return probs
    