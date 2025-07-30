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


# Data Science Libs
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from textblob import TextBlob
from sklearn.metrics.pairwise import cosine_similarity

# Chatbot
import google.generativeai as genai


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
data = pd.read_csv(f"{root_directory}/data/customer_data.csv")
products = ["Laptop", "Smartphone", "Tablet", "Headphone", "Smartwatch", "Kamera", "Speaker", "Mouse", "Keyboard"]


# Preprocessing: Standarisasi Data
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data[["Age", "Income", "SpendingScore"]])

# K-Means Clustering
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
data["Cluster"] = kmeans.fit_predict(data_scaled)

# **Loyalty Score** → Skor berdasarkan jumlah transaksi & nilai total pembelian
data["LoyaltyScore"] = (data["SpendingScore"] + (data["Income"] / 1000)) / 2

# **Customer Lifetime Value (CLV)** → Menghitung CLV sederhana
data["CLV"] = data["SpendingScore"] * data["Income"] / 1000

# **Prediksi Churn**
# Simulasi: Anggap pelanggan dengan Spending Score < 30 memiliki risiko churn tinggi
data["Churn"] = (data["SpendingScore"] < 30).astype(int)

# Model Prediksi Churn
X = data[["Age", "Income", "SpendingScore", "LoyaltyScore", "CLV"]]
y = data["Churn"]
churn_model = xgb.XGBClassifier(objective="binary:logistic", base_score=0.5)
churn_model.fit(X, y)




#  INFORMATION XTRACTOR
# Load model NER dari Hugging Face
# nlp = pipeline("ner", model="dbmdz/bert-base-cased-finetuned-conll03-english")
# nlp = pipeline("ner", model="dslim/distilbert-NER")
# nlp = pipeline("ner", from_tf=True, model="aadhistii/DistilBERT-Indonesian-NER")
# nlp = pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english")
# model_name = "gagan3012/bert-tiny-finetuned-ner"

from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification
import streamlit as st

# Load model dan tokenizer
@st.cache_resource
def load_model():
    model_name = "dslim/distilbert-NER"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # model = AutoModelForTokenClassification.from_pretrained(model_name)
    nlp = pipeline("ner", model=model_name)
    return nlp

nlp = load_model()



# Chatbot Init Geminni Model
genai.configure(api_key=GEMINI_API_KEY) 

# --- Tambahan: Dapatkan daftar model yang tersedia ---
available_models = []
try:
    for m in genai.list_models():
        # Hanya sertakan model yang mendukung generateContent
        if 'generateContent' in m.supported_generation_methods:
            available_models.append(m.name)
    print(f"Available models for generateContent: {available_models}")

    if not available_models:
        raise ValueError("No models found that support 'generateContent'. Check your API key and region.")

    # Gunakan model pertama yang ditemukan atau pilih yang spesifik
    # Contoh: 'gemini-1.5-flash' atau 'gemini-1.0-pro'
    # Ganti dengan nama model yang paling sesuai untuk kebutuhan Anda dari daftar di atas.
    # Misalnya, jika Anda melihat 'models/gemini-1.5-flash', gunakan itu.
    MODEL_TO_USE = "models/gemini-2.5-flash-lite" # Mengambil model pertama yang didukung
    # Atau secara eksplisit: MODEL_TO_USE = "models/gemini-1.5-flash"
    print(f"Using model: {MODEL_TO_USE}")

except Exception as e:
    print(f"Error listing models: {e}")
    # Jika tidak dapat membuat model karena ini, set model ke None untuk penanganan error lebih lanjut
    model = None
    # Exit atau tangani secara berbeda jika ini adalah error fatal untuk aplikasi Anda
    raise e # Re-raise error jika tidak ada model yang dapat ditemukan

model = genai.GenerativeModel(MODEL_TO_USE)


# from transformers import AutoModel, AutoTokenizer, pipeline
# model_name = "aadhistii/DistilBERT-Indonesian-NER"

# model = AutoModel.from_pretrained(model_name, from_tf=True)
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# nlp = pipeline(tokenizer, model=f"{model}")

if __name__ == '__main__':
    app.run()

