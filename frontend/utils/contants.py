import os
from dotenv import load_dotenv
load_dotenv()

BASE_URL = os.getenv("BASE_URL")
OPENAI_API_KEY=os.getenv("OPENAI_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GOOGLE_PLACES_URL = os.getenv("GOOGLE_PLACES_URL")
DB_URL = os.getenv("DB_URL", "sqlite:///data/companies.db")
GEMINI_API_KEY=os.getenv("GEMINI_API_KEY")