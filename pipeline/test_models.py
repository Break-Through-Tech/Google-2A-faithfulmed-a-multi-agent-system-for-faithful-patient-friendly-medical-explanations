import os
import time
from dotenv import load_dotenv
from google import genai

load_dotenv()
client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

for model in [
    "gemini-3.1-flash-lite",
    "gemini-3.5-flash-lite",
    "gemini-3.5-flash",
    "gemini-3.7-flash",
    "gemini-3.8-flash",
]:
    start = time.time()
    try:
        client.models.generate_content(model=model, contents="Say hi")
        print(f"{model}: OK  {time.time() - start:.1f}s")
    except Exception as e:
        print(f"{model}: FAIL {time.time() - start:.1f}s  {str(e)[:80]}")