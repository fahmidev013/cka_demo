import os
from flask import Flask, request, jsonify, send_file
from frontend.utils.contants import BASE_URL, OPENAI_API_KEY, GOOGLE_API_KEY
from dotenv import load_dotenv
from flask_cors import CORS
from langchain_community.llms import OpenAI  
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
import pandas as pd

load_dotenv()

app = Flask(__name__)

# Untuk mengizinkan komunikasi dengan frontend
CORS(app)

llm = OpenAI()
conversation = ConversationChain(llm=llm, memory=ConversationBufferMemory())


if __name__ == '__main__':
    app.run()

