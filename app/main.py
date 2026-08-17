from fastapi import FastAPI

from app.api.chat import router as chat_router
from app.api.document import router as document_router
from app.api.image import router as image_router
from app.api.project import router as project_router
from app.api.upload import router as upload_router


app = FastAPI(
    title="Chat Platform",
    description="Enterprise Document Intelligence Platform",
    version="1.0.0",
)


app.include_router(chat_router)
app.include_router(upload_router)
app.include_router(document_router)
app.include_router(project_router)
app.include_router(image_router)


@app.get("/")
def root():
    return {
        "message": "Chat Platform API is running."
    }


@app.get("/health")
def health():
    return {
        "status": "healthy"
    }