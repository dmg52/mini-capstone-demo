# app/db.py
import os
from dotenv import load_dotenv
from neo4j import GraphDatabase

load_dotenv()

_URI  = os.getenv("NEO4J_URI")
_USER = os.getenv("NEO4J_USER")
_PWD  = os.getenv("NEO4J_PASS")

if not all([_URI, _USER, _PWD]):
    raise RuntimeError("Missing Neo4j connection info")

# this is your singleton driver everyone shares
driver = GraphDatabase.driver(_URI, auth=(_USER, _PWD))
