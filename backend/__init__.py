import os
from .functions import *
from flask import Flask, request, jsonify, send_file
from frontend.utils.contants import *
from frontend.utils.functions import *
from dotenv import load_dotenv
from flask_cors import CORS
from langchain_community.llms import OpenAI  
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory

import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import time
from sqlalchemy import Column, Integer, Float, String, Text, create_engine
from sqlalchemy.orm import declarative_base, sessionmaker
from data.models import Company, get_engine_and_session

# from transformers import pipeline

load_dotenv()

app = Flask(__name__)

# Untuk mengizinkan komunikasi dengan frontend
CORS(app)

llm = OpenAI()
conversation = ConversationChain(llm=llm, memory=ConversationBufferMemory())


# Ambil DB dari .env atau default ke SQLite
# engine, db_session = get_engine_and_session(DB_URL)


# Activate Data Science Page

# Load dataset
root_directory = os.getcwd()



#  INFORMATION XTRACTOR
# Load model NER dari Hugging Face
# nlp = pipeline("ner", model="dbmdz/bert-base-cased-finetuned-conll03-english")
# nlp = pipeline("ner", model="dslim/distilbert-NER")
# nlp = pipeline("ner", from_tf=True, model="aadhistii/DistilBERT-Indonesian-NER")
# nlp = pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english")


from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification
import streamlit as st

# Load model dan tokenizer
@st.cache_resource
def load_model():
    model_name = "gagan3012/bert-tiny-finetuned-ner"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # model = AutoModelForTokenClassification.from_pretrained(model_name)
    nlp = pipeline("ner", model=model_name, aggregation_strategy="simple")
    return nlp

nlp = load_model()



# from transformers import AutoModel, AutoTokenizer, pipeline
# model_name = "aadhistii/DistilBERT-Indonesian-NER"

# model = AutoModel.from_pretrained(model_name, from_tf=True)
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# nlp = pipeline(tokenizer, model=f"{model}")

if __name__ == '__main__':
    app.run()

