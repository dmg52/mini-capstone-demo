# app/openai_client.py
import os
from dotenv import load_dotenv
from openai import OpenAI

# 1) Load .env (so OPENAI_API_KEY is available)
load_dotenv()

API_KEY = os.getenv("OPENAI_API_KEY")
if not API_KEY:
    raise RuntimeError("Missing OPENAI_API_KEY in environment")

# 2) Create a single shared client
client = OpenAI(api_key=API_KEY)
