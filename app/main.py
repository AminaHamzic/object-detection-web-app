from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path

app = FastAPI()

# Define the path to your frontend directory
frontend_dir = Path(__file__).resolve().parent.parent / 'frontend'

# Serve static files from the 'frontend' directory
app.mount("/static", StaticFiles(directory=frontend_dir), name="static")


@app.get("/")
def read_root():
    return FileResponse(frontend_dir / 'index.html')


@app.get("/hello/{name}")
async def say_hello(name: str):
    return {"message": f"Hello {name}"}
