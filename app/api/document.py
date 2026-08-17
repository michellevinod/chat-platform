from fastapi import APIRouter, HTTPException, Query

from app.repositories.qdrant_repository import QdrantRepository


router = APIRouter(
    prefix="/documents",
    tags=["Documents"],
)


_repo = QdrantRepository()


@router.get("")
def list_documents(
    project_name: str | None = Query(
        default=None,
        description="Filter documents by project name",
    ),
):
    """
    List all documents stored in the vector database.
    """

    documents = _repo.get_distinct_documents(
        project_name=project_name,
    )

    return {
        "success": True,
        "documents": documents,
        "count": len(documents),
    }


@router.delete("")
def delete_document(
    project_name: str = Query(
        ...,
        description="Project containing the document",
    ),
    document_name: str = Query(
        ...,
        description="Document to delete",
    ),
):
    """
    Delete one document from Qdrant.

    Deletion is scoped to both project_name and document_name.
    This prevents a document with the same name in another project
    from being deleted.
    """

    project_name = project_name.strip()
    document_name = document_name.strip()

    if not project_name:
        raise HTTPException(
            status_code=400,
            detail="project_name cannot be empty.",
        )

    if not document_name:
        raise HTTPException(
            status_code=400,
            detail="document_name cannot be empty.",
        )

    _repo.delete_document(
        collection_name="documents",
        project_name=project_name,
        document_name=document_name,
    )

    return {
        "success": True,
        "message": "Document deleted successfully.",
        "project": project_name,
        "document": document_name,
    }