from fastapi import APIRouter, Query
from app.repositories.qdrant_repository import QdrantRepository

router = APIRouter(
    prefix="/documents",
    tags=["Documents"],
)

_repo = QdrantRepository()


@router.get("")
def list_documents(
    project_name: str | None = Query(default=None, description="Filter documents by project name"),
):
    """
    List all documents stored in the vector database.
    """
    documents = _repo.get_distinct_documents(project_name=project_name)
    return {
        "success": True,
        "documents": documents,
        "count": len(documents),
    }
