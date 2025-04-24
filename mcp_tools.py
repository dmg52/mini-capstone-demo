import json
import os
import re
from openai import OpenAI
from mcp.server.fastmcp import FastMCP

# 1) Instantiate the MCP server
mcp = FastMCP("Ingestion MCP")

# 2) OpenAI client (share or re-init here)
TEST_MODE = os.getenv("TEST_MODE", "false").lower() in ("1","true","yes")

if not TEST_MODE:
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# 3) Cleaning tool
@mcp.tool()
def clean_text(raw: str) -> str:
    text = raw.lower()
    text = re.sub(r"\s+", " ", text).strip()
    return "".join(ch for ch in text if ch.isprintable())

# 4) Entity extraction tool
@mcp.tool()
def extract_entities(text: str) -> list[dict]:
    """
    Return a list of {name, type} dicts from 'text'.
    """
    if TEST_MODE:
        return []
    
    prompt = f"""
Extract all unique named entities from the following text.

• Return _only_ a JSON array of objects with exactly two keys:
  – "name": the exact entity string (in lowercase)
  – "type": the entity category (e.g., PERSON, ORG, LOCATION, DATE, etc.)

Do not include any explanatory text, markdown fences, or code blocks—just the JSON.
If there are no entities to be found, simply return an empty JSON.
Text:

{text}
"""
    resp = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
    )
    raw = resp.choices[0].message.content
    content = re.sub(r"```(?:json)?\s*(\[.*?\])\s*```", r"\1", raw, flags=re.S)
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        raise ValueError(f"Could not parse entities JSON:\n{content}")

# 5) Relation extraction tool
@mcp.tool()
def extract_relations(text: str) -> list[dict]:
    if TEST_MODE:
        return []
    
    prompt = f"""
Extract all relationships from the following text.  
– Output must be a JSON array of objects, each with keys:
  • subject  
  • predicate  
  • object  

**Return _only_ the JSON array.** No narrative, no markdown, no code fences. Read and extract all entities and relationships as lower-case.
If there are no relationships to be found, simply return an empty JSON.
Text:

{text}
"""
    resp = client.chat.completions.create(
        model="gpt-4o", 
        messages=[{"role":"user","content":prompt}],
    )
    raw = resp.choices[0].message.content
    content = re.sub(r"```(?:json)?\s*(\[.*?\])\s*```", r"\1", raw, flags=re.S)
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        raise ValueError(f"Could not parse entities JSON:\n{content}")

# 6) Embedding tool
@mcp.tool()
def embed_text(text: str) -> list[float]:
    if TEST_MODE:
        # return a fixed dummy vector
        return [0.0] * 1536
    
    resp = client.embeddings.create(input=text, model="text-embedding-ada-002")
    return resp.data[0].embedding
