from fastapi import APIRouter
from app.models.schema import QueryRequest
from app.services.query_service import handle_query, cache

router = APIRouter()


@router.post("/query")
def query(req: QueryRequest):

    return handle_query(req.query)


@router.get("/cache/stats")
def cache_stats():

    return cache.stats()


@router.delete("/cache")
def clear_cache():

    cache.clear()

    return {"message": "cache cleared"}