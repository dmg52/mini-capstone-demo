from fastapi import APIRouter, File, UploadFile, HTTPException
from pathlib import Path

from app.mcp_tools import clean_text
from app.routers.upload import parse_document

router = APIRouter(prefix="/test", tags=["MCP Tools Tests"])

# File upload ingestion 
@router.post("/test/clean")
async def test_clean_pdf(file: UploadFile = File(...)):
    # 1) Check extension
    suffix = Path(file.filename).suffix.lower()
    if suffix != ".pdf":
        raise HTTPException(400, "Only PDF files are supported.")

    # 2) Save to temp
    tmp_path = f"/tmp/{file.filename}"
    with open(tmp_path, "wb") as f:
        f.write(await file.read())

    # 3) Extract raw text
    try:
        raw_text = parse_document(tmp_path, suffix)
    except Exception as e:
        raise HTTPException(500, f"PDF parsing failed: {e}")

    # 4) Clean via your MCP tool
    cleaned = clean_text(raw_text)

    # 5) Return cleaned text
    return {"cleaned_text": cleaned}