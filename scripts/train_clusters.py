import numpy as np
import joblib
from sklearn.mixture import GaussianMixture

embeddings = np.load("storage/embeddings.npy")

gmm = GaussianMixture(
    n_components=30,
    covariance_type="tied",
    random_state=42
)

gmm.fit(embeddings)

joblib.dump(gmm, "storage/gmm.pkl")

print("Cluster model saved")