from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses   import FileResponse
from pathlib import Path

from app.database import driver
from app.mcp_tools import mcp
from app.openai_client import client
from app.routers import upload, testrouter

def get_some_data(tx, limit: int = 5):
    query = "MATCH (n) RETURN n LIMIT $limit"
    res = tx.run(query, limit=limit)
    return [record["n"] for record in res]

#  Create FastAPI App
app = FastAPI(title="FastAPI + Neo4j + OpenAI + MCP Demo")

# Mount the MCP server under /mcp
app.mount("/mcp", mcp.sse_app())

# Mount static file
app.mount("/static", StaticFiles(directory="static", html=True), name="static")
app.mount("/images", StaticFiles(directory="images"), name="images")

# Inlude Routers
app.include_router(testrouter.router)
app.include_router(upload.router)

@app.get("/", response_class=FileResponse)
def serve_spa():
    return Path("static/index.html")

# Simple health-check
@app.get("/health")
async def health():
    return {"status": "ok"}

# Demo Neo4j endpoint
@app.get("/neo4j-nodes")
async def read_nodes(limit: int = 5):
    with driver.session() as session:
        nodes = session.read_transaction(get_some_data, limit)
    return {"nodes": nodes}

def _wipe_db(tx):
    tx.run("MATCH (n) DETACH DELETE n")

# ─── Endpoint to wipe the entire Neo4j graph ────────────────────────────
@app.delete("/clear-neo4j", summary="Delete all nodes & relationships")
async def wipe_database():
    """
    WARNING: This will irreversibly delete every node and relationship in Neo4j.
    """
    try:
        with driver.session() as session:
            session.write_transaction(_wipe_db)
        return {"status": "ok", "message": "All nodes and relationships deleted."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to wipe database: {e}")

# Demo OpenAI endpoint
@app.post("/chat")
async def chat(prompt: str):
    try:
        resp = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
        )
        return {"reply": resp.choices[0].message.content}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
