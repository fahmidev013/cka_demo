import os
from dotenv import load_dotenv
load_dotenv()

BASE_URL = os.getenv("BASE_URL")
OPENAI_API_KEY=os.getenv("OPENAI_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")