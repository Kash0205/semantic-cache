from sentence_transformers import SentenceTransformer
from app.config import MODEL_NAME

model = SentenceTransformer(MODEL_NAME)


def embed_text(text):
    return model.encode(text)


def embed_batch(texts):
    return model.encode(texts, show_progress_bar=True)