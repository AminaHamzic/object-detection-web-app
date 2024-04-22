from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path
from fastapi.middleware.cors import CORSMiddleware

UPLOAD_DIR = Path() / 'uploads'

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # temporary
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

frontend_dir = Path(__file__).resolve().parent.parent / 'frontend'
app.mount("/static", StaticFiles(directory=frontend_dir), name="static")
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")


@app.post("/uploadfile/")
async def create_upload_file(file_upload: UploadFile):
    data = await file_upload.read()
    save_to = UPLOAD_DIR / file_upload.filename
    with open(save_to, 'wb') as f:
        f.write(data)
    return {"filename": file_upload.filename, "url": f"/uploads/{file_upload.filename}"}


@app.get("/")
def read_root():
    return FileResponse(frontend_dir / 'index.html')


@app.get("/hello/{name}")
async def say_hello(name: str):
    return {"message": f"Hello {name}"}
