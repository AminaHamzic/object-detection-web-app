from fastapi import APIRouter

router = APIRouter()


@router.get("/detect/")
async def detect_object():
    return {"message": "Object detected"}
