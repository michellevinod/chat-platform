from __future__ import annotations

import json
import os
import uuid
from pathlib import Path

from app.repositories.qdrant_repository import QdrantRepository


class ProjectRepository:
    """
    Repository for project management.

    Project names are persisted in storage/projects.json so that
    empty projects can exist before any document is uploaded.

    Existing projects already present in Qdrant are automatically
    discovered and preserved.
    """

    def __init__(self) -> None:
        self._qdrant = QdrantRepository()

        self._storage_dir = Path(
            os.getenv(
                "PROJECT_STORAGE_DIR",
                "storage",
            )
        )

        self._file = self._storage_dir / "projects.json"

        self._storage_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        self._ensure_file()

    # ================================================================
    # INTERNAL STORAGE
    # ================================================================

    def _ensure_file(self) -> None:
        if not self._file.exists():
            self._write(
                {
                    "projects": []
                }
            )

    def _read(self) -> dict:
        try:
            with self._file.open(
                "r",
                encoding="utf-8",
            ) as file:
                data = json.load(file)

            if not isinstance(data, dict):
                return {"projects": []}

            if not isinstance(
                data.get("projects"),
                list,
            ):
                data["projects"] = []

            return data

        except (
            OSError,
            json.JSONDecodeError,
        ):
            return {"projects": []}

    def _write(self, data: dict) -> None:
        temp_file = self._file.with_suffix(
            ".tmp"
        )

        with temp_file.open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                data,
                file,
                indent=2,
            )

        temp_file.replace(self._file)

    # ================================================================
    # PROJECT LIST
    # ================================================================

    def list_projects(self) -> list[str]:
        data = self._read()

        registry_names = {
            str(project.get("name")).strip()
            for project in data["projects"]
            if isinstance(project, dict)
            and project.get("name")
        }

        # Preserve projects created before the registry existed.
        qdrant_names = set(
            self._qdrant.get_distinct_projects()
        )

        all_names = (
            registry_names
            | qdrant_names
        )

        # Bring discovered Qdrant projects into the registry.
        existing_names = {
            str(project.get("name")).strip()
            for project in data["projects"]
            if isinstance(project, dict)
            and project.get("name")
        }

        changed = False

        for name in sorted(qdrant_names):
            if name not in existing_names:
                data["projects"].append(
                    {
                        "id": f"project_{uuid.uuid4().hex[:12]}",
                        "name": name,
                    }
                )
                changed = True

        if changed:
            self._write(data)

        return sorted(all_names)

    # ================================================================
    # CREATE
    # ================================================================

    def create_project(
        self,
        name: str,
    ) -> dict:
        name = name.strip()

        if not name:
            raise ValueError(
                "Project name cannot be empty."
            )

        projects = self.list_projects()

        if name.lower() in {
            project.lower()
            for project in projects
        }:
            raise ValueError(
                f'Project "{name}" already exists.'
            )

        data = self._read()

        project = {
            "id": f"project_{uuid.uuid4().hex[:12]}",
            "name": name,
        }

        data["projects"].append(project)

        self._write(data)

        return project

    # ================================================================
    # RENAME
    # ================================================================

    def rename_project(
        self,
        old_name: str,
        new_name: str,
    ) -> dict:
        old_name = old_name.strip()
        new_name = new_name.strip()

        if not old_name:
            raise ValueError(
                "Current project name cannot be empty."
            )

        if not new_name:
            raise ValueError(
                "New project name cannot be empty."
            )

        if old_name == new_name:
            raise ValueError(
                "New project name is the same as the current name."
            )

        projects = self.list_projects()

        if old_name not in projects:
            raise ValueError(
                f'Project "{old_name}" does not exist.'
            )

        if new_name.lower() in {
            project.lower()
            for project in projects
            if project.lower() != old_name.lower()
        }:
            raise ValueError(
                f'Project "{new_name}" already exists.'
            )

        data = self._read()

        found = False

        for project in data["projects"]:
            if (
                isinstance(project, dict)
                and str(
                    project.get("name", "")
                ).strip()
                == old_name
            ):
                project["name"] = new_name
                found = True
                break

        if not found:
            data["projects"].append(
                {
                    "id": f"project_{uuid.uuid4().hex[:12]}",
                    "name": new_name,
                }
            )

        self._write(data)

        # Update every indexed chunk belonging to the project.
        self._qdrant.rename_project(
            old_name=old_name,
            new_name=new_name,
        )

        return {
            "old_name": old_name,
            "new_name": new_name,
        }

    # ================================================================
    # DELETE
    # ================================================================

    def delete_project(
        self,
        name: str,
    ) -> None:
        name = name.strip()

        if not name:
            raise ValueError(
                "Project name cannot be empty."
            )

        projects = self.list_projects()

        if name not in projects:
            raise ValueError(
                f'Project "{name}" does not exist.'
            )

        data = self._read()

        data["projects"] = [
            project
            for project in data["projects"]
            if not (
                isinstance(project, dict)
                and str(
                    project.get("name", "")
                ).strip()
                == name
            )
        ]

        self._write(data)

        # Remove all indexed chunks belonging to this project.
        self._qdrant.delete_project(
            project_name=name,
        )