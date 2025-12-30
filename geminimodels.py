import os
import dotenv
from google import genai

dotenv.load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise RuntimeError("GEMINI_API_KEY not found")

client = genai.Client(api_key=api_key)

print("\nAvailable Gemini models for this API key:\n")

models = list(client.models.list())

for m in models:
    print(f"Model name: {m.name}")
    # Print whatever metadata is safely available
    print(f"  Display name: {getattr(m, 'display_name', 'N/A')}")
    print(f"  Description : {getattr(m, 'description', 'N/A')}")
    print("-" * 60)

print(f"\nTotal models visible: {len(models)}")
