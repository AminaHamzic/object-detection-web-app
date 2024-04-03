from fastapi import FastAPI, UploadFile
from pathlib import Path

app = FastAPI()


app.include_router(detection.router)
app.include_router(upload.router)

""""
@app.get("/")
async def root():
    return {"message": "Hello Amina"}


@app.get("/hello/{name}")
async def say_hello(name: str):
    return {"message": f"Hello {name}"}


@app.post('/uploadfile/')
async def create_upload_file(file_upload: UploadFile):
    return {"filename": file_upload.filename} """