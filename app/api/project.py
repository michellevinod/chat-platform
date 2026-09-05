from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.repositories.project_repository import ProjectRepository


router = APIRouter(
    prefix="/projects",
    tags=["Projects"],
)


_repo = ProjectRepository()


class CreateProjectRequest(BaseModel):
    name: str = Field(
        min_length=1,
        max_length=200,
    )


class RenameProjectRequest(BaseModel):
    old_name: str = Field(
        min_length=1,
        max_length=200,
    )

    new_name: str = Field(
        min_length=1,
        max_length=200,
    )


@router.get("")
def list_projects():
    """
    List all projects.
    """
    projects = _repo.list_projects()

    return {
        "success": True,
        "projects": projects,
        "count": len(projects),
    }


@router.post("")
def create_project(
    request: CreateProjectRequest,
):
    try:
        project = _repo.create_project(
            request.name
        )

        return {
            "success": True,
            "project": project,
        }

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        )


@router.patch("")
def rename_project(
    request: RenameProjectRequest,
):
    try:
        result = _repo.rename_project(
            old_name=request.old_name,
            new_name=request.new_name,
        )

        return {
            "success": True,
            **result,
        }

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        )


@router.delete("")
def delete_project(
    name: str,
):
    try:
        _repo.delete_project(name)

        return {
            "success": True,
            "project": name,
        }

    except ValueError as exc:
        raise HTTPException(
            status_code=404,
            detail=str(exc),
        )