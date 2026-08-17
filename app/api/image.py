from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse


router = APIRouter(
    prefix="/images",
    tags=["Images"],
)


IMAGE_DIRECTORY = Path("storage/images").resolve()


@router.get("/{image_name}")
def get_image(
    image_name: str,
):
    """
    Serve an extracted document image.

    Only files inside storage/images are allowed.
    """

    image_path = (
        IMAGE_DIRECTORY / image_name
    ).resolve()

    # Prevent path traversal.
    if IMAGE_DIRECTORY not in image_path.parents:
        raise HTTPException(
            status_code=400,
            detail="Invalid image path.",
        )

    if not image_path.is_file():
        raise HTTPException(
            status_code=404,
            detail="Image not found.",
        )

    allowed_extensions = {
        ".png",
        ".jpg",
        ".jpeg",
        ".webp",
    }

    if image_path.suffix.lower() not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail="Unsupported image type.",
        )

    return FileResponse(
        path=image_path,
    )