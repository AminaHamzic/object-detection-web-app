import boto3
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path
from botocore.exceptions import NoCredentialsError
from starlette.responses import JSONResponse

app = FastAPI()
s3_client = boto3.client('s3')


frontend_dir = Path(__file__).resolve().parent.parent / 'frontend'
app.mount("/static", StaticFiles(directory=frontend_dir), name="static")


@app.get("/")
def read_root():
    return FileResponse(frontend_dir / 'index.html')


@app.get("/hello/{name}")
async def say_hello(name: str):
    return {"message": f"Hello {name}"}


@app.post("/uploadfile/")
async def upload_file_to_s3(file: UploadFile = File(...)):
    file_contents = await file.read()
    try:
        s3_client.put_object(
            Bucket="uplodefilebucket",
            Key=file.filename,
            Body=file_contents
        )
        return JSONResponse(content={"message": "Upload successful"}, status_code=200)
    except Exception as e:
        # Handle exception
        return JSONResponse(content={"message": str(e)}, status_code=500)