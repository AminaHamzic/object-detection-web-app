from fastapi import APIRouter, UploadFile, File
from ..utils import s3_utils

router = APIRouter()


@router.post("/uploadfile/")
async def create_upload_file(file: UploadFile = File(...)):
    # Your upload logic here, potentially using s3_utils
    return {"filename": file.filename}

"""
@app.post('/uploadfile/')
async def create_upload_file(file_upload: UploadFile):
    return {"filename": file_upload.filename} """