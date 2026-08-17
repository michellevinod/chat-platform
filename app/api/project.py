from fastapi import APIRouter
from app.repositories.qdrant_repository import QdrantRepository

router = APIRouter(
    prefix="/projects",
    tags=["Projects"],
)

_repo = QdrantRepository()


@router.get("")
def list_projects():
    """
    List all projects stored in the vector database.
    """
    projects = _repo.get_distinct_projects()
    return {
        "success": True,
        "projects": projects,
        "count": len(projects),
    }
