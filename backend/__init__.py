import os
from flask import Flask, request, jsonify, send_file
from frontend.utils.contants import *
from dotenv import load_dotenv
from flask_cors import CORS
from langchain_community.llms import OpenAI  
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
import pandas as pd
import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import time
from sqlalchemy import Column, Integer, Float, String, Text, create_engine
from sqlalchemy.orm import declarative_base, sessionmaker
from data.models import Company, get_engine_and_session

load_dotenv()

app = Flask(__name__)

# Untuk mengizinkan komunikasi dengan frontend
CORS(app)

llm = OpenAI()
conversation = ConversationChain(llm=llm, memory=ConversationBufferMemory())


# Ambil DB dari .env atau default ke SQLite
engine, db_session = get_engine_and_session(DB_URL)

if __name__ == '__main__':
    app.run()

