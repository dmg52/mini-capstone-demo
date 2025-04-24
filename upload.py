from fastapi import APIRouter, File, UploadFile, BackgroundTasks, HTTPException
from pdfminer.high_level import extract_text
import os
import re
# import yt_dlp
# import whisper
import tempfile
from openai import OpenAI
from pydantic import BaseModel
from urllib.parse import urlparse, parse_qs

from app.database import driver
from app.mcp_tools import clean_text, extract_entities, extract_relations, embed_text

router = APIRouter(prefix="/upload", tags=["Upload"])

# _whisper_model = whisper.load_model("base")
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

'''def transcribe_youtube_video(youtube_url: str) -> str:
    # Create a unique temp file for audio
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
        tmp_path = tmp.name

    # Download & extract audio to that temp file
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": tmp_path,
        "postprocessors": [{
            "key": "FFmpegExtractAudio",
            "preferredcodec": "mp3",
            "preferredquality": "192",
        }],
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([youtube_url])

    try:
        # Transcribe with Whisper
        result = _whisper_model.transcribe(tmp_path)
        return result["text"]
    finally:
        # 5) Always clean up the temp file
        try:
            os.remove(tmp_path)
        except OSError:
            pass'''

def parse_document(path: str, doc_type: str) -> str:
    """
    Return raw text from a file.
    Currently supports PDFs via pdfminer.six and YouTube urls.
    """
    ext = doc_type.lower()
    if ext == ".pdf":
        try:
            # extract_text will process the PDF and return all its text
            text = extract_text(path)
            return text
        except Exception as e:
            raise RuntimeError(f"Error parsing PDF {path!r}: {e}")
    elif ext == ".youtube":
        try:
            # text = transcribe_youtube_video(path)
            text = ""
            return text
        except Exception as e:
            raise RuntimeError(f"Error parsing Youtube URL {path!r}: {e}")
    else:
        raise ValueError(f"Unsupported doc_type: {doc_type!r}")
    
'''def _store_document(tx, doc_id: str, text: str, embedding: list[float]):
    tx.run(
        """
        MERGE (d:Document {id: $doc_id})
        SET d.text = $text, d.embedding = $embedding
        """,
        doc_id=doc_id,
        text=text,
        embedding=embedding,
    )'''

def _sanitize_label(name: str) -> str:
    # replace any non-word characters with underscore
    return re.sub(r'\W+', '_', name)

def _store_entity(tx, name: str, etype: str, embedding: list[float], doc_id: str):
    # Sanitize the entity name into a valid label
    safe_label = _sanitize_label(name)
    cypher = f"""
    MERGE (e:Entity:`{safe_label}` {{ name: $name, type: $etype }})
    SET
      e.text = $name,
      e.embedding = $embedding,
      e.doc_id = $doc_id
    """

    tx.run(
        cypher,
        name=name,
        etype=etype,
        embedding=embedding,
        doc_id=doc_id,
    )

def _store_relation(tx, subject: str, predicate: str, object_name: str):
    rel_label = _sanitize_label(predicate)
    cypher = f"""
    MATCH (s:Entity {{ name: $subject }}),
          (o:Entity {{ name: $object }})
    MERGE (s)-[r:`{rel_label}` {{ type: $predicate }}]->(o)
    """

    tx.run(
        cypher,
        subject=subject,
        object=object_name,
        predicate=predicate,
    )

def ingest_pipeline(text: str, doc_id: str):
    """
    1) Clean text
    2) Extract entities & relations
    3) Generate embeddings
    4) Write to Neo4j
    """
    cleaned   = clean_text(text)
    entities  = extract_entities(cleaned)
    relations = extract_relations(cleaned)
    entity_embeddings = {
        ent["name"]: embed_text(ent["name"])
        for ent in entities
    }

    # Persist into Neo4j
    with driver.session() as session:
        # Upsert the Document node
        # session.write_transaction(_store_document, doc_id, cleaned, embedding)

        # Upsert Entity nodes
        for ent in entities:
            name = ent["name"]
            etype = ent["type"]
            emb = entity_embeddings[name]
            session.write_transaction(_store_entity, name, etype, emb, doc_id)

        # Create relationship edges
        for rel in relations:
            session.write_transaction(
                _store_relation,
                rel["subject"],
                rel["predicate"],
                rel["object"],
            )
    

# File upload ingestion 
@router.post("/ingest/file")
async def ingest_file(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
):
    # save to a temp file
    suffix = os.path.splitext(file.filename)[1]
    tmp_path = f"/tmp/{file.filename}"
    with open(tmp_path, "wb") as f:
        f.write(await file.read())

    # kick off the ingestion pipeline in the background
    doc_id = file.filename  # or generate UUID
    background_tasks.add_task(ingest_pipeline, parse_document(tmp_path, suffix), doc_id)

    return {"status": "queued", "document": file.filename}

class YouTubeURL(BaseModel):
    url: str

@router.post("/ingest/youtube", summary="Upload a YouTube URL for ingestion")
async def ingest_youtube(
    background_tasks: BackgroundTasks,
    payload: YouTubeURL,
):
    # 1) Extract a document ID (here we use the video ID)
    parsed = urlparse(payload.url)
    if parsed.hostname in ("youtu.be",):
        video_id = parsed.path.lstrip("/")
    elif parsed.hostname in ("www.youtube.com", "youtube.com"):
        qs = parse_qs(parsed.query)
        video_id = qs.get("v", [None])[0]
    else:
        raise HTTPException(400, "Unsupported YouTube URL")

    if not video_id:
        raise HTTPException(400, "Could not parse video ID from URL")

    # 2) Parse/transcribe the video (this may block briefly)
    try:
        raw_text = parse_document(payload.url, ".youtube")
    except Exception as e:
        raise HTTPException(500, f"Transcription failed: {e}")

    # 3) Queue the full ingestion (clean → extract → embed → persist)
    background_tasks.add_task(ingest_pipeline, raw_text, video_id)

    return {
        "status": "queued",
        "video_id": video_id,
        "message": "Video transcription started and queued for ingestion"
    }

def fetch_entity_context(entity_name: str, depth: int = 3) -> str:
    """
    Pull all relationships up to `depth` hops away from the given entity,
    and serialize them into plain-text SPO triples (one per line).
    """
    bound = f"1..{depth}"

    cypher = f"""
    MATCH (e)-[r*{bound}]-(o)
    WHERE toLower(e.name) = toLower($name)
    UNWIND r AS rel
    WITH e, rel, o
    RETURN e.name    AS subject,
           rel.type  AS predicate,
           o.name    AS object
    """

    with driver.session() as session:
        result = session.run(cypher, name=entity_name)

        lines = []
        for rec in result:
            subj = rec["subject"]
            pred = rec["predicate"]
            obj  = rec["object"]
            lines.append(f"{subj} {pred} {obj}")
    
    return "\n".join(lines)

class QARequest(BaseModel):
    question: str
    #focus:   str  # the entity or keyword to center your query on

@router.post("/qa")
async def answer_question(req: QARequest):
    # 1) Fetch your graph context
    entities = extract_entities(req.question)
    if not entities:
        raise HTTPException(404, "No entities found in question")
    
    focus = entities[0]["name"]

    context = fetch_entity_context(focus, depth=3)
    if not context:
        raise HTTPException(404, f"No context found for entity {focus!r}")

    # 2) Build a prompt that injects the graph facts
    prompt = f"""
You have the following facts from a knowledge graph, one per line:
{context}

Answer the question below using only these facts. If you don't know, give any information about the subject and object that you can. Otherwise, say you do not have enough information.
Do not restate  the given context in the beginning of the response.

Question: {req.question}
"""

    # 3) Call the LLM
    try:
        resp = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role":"system","content":"You are a precise, helpful AI assistant."},
                      {"role":"user","content":prompt}],
        )
        return {"answer": resp.choices[0].message.content.strip()}
    except Exception as e:
        raise HTTPException(500, detail=str(e))