Project Overview
This project implements a semantic search system using:

• SentenceTransformer embeddings
• FAISS vector database
• Gaussian Mixture fuzzy clustering
• Custom semantic cache
• FastAPI service

System Architecture
Query
 ↓
Embedding
 ↓
Semantic Cache
 ↓
Vector DB Search (if miss)
 ↓
Return results

Running Instructions
python -m venv venv
pip install -r requirements.txt

python -m scripts.preprocess
python -m scripts.build_embeddings
python -m scripts.build_index
python -m scripts.train_clusters

uvicorn app.main:app --reload

API Endpoints
POST /query
GET /cache/stats
DELETE /cache

Docker

Run:

docker build -t semantic-cache .
docker run -p 8000:8000 semantic-cache
