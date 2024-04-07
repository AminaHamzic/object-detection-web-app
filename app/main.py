from fastapi import FastAPI
from fastapi.responses import FileResponse
from pathlib import Path

app = FastAPI()

# Assuming __file__ is in the 'app' directory, and the 'frontend' directory is at the same level
root_path = Path(__file__).resolve().parent.parent
frontend_dir = root_path / 'frontend'


@app.get("/")
def read_root():
    return FileResponse(frontend_dir / 'index.html')


@app.get("/hello/{name}")
async def say_hello(name: str):
    return {"message": f"Hello {name}"}