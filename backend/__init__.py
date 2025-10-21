import os
from .functions import *
from flask import Flask, request, jsonify, send_file
from frontend.utils.contants import *
from dotenv import load_dotenv
from flask_cors import CORS

import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import time
import gc
from sqlalchemy import Column, Integer, Float, String, Text, create_engine
from sqlalchemy.orm import declarative_base, sessionmaker
from data.models import Company, get_engine_and_session


# Data Science Libs
import threading
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
from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification

load_dotenv()

app = Flask(__name__)

# Untuk mengizinkan komunikasi dengan frontend
CORS(app)

@app.after_request
def _cleanup_after_request(response):
    # Garbage collect after every request to free unused memory
    try:
        gc.collect()
    except Exception:
        pass
    return response

@app.teardown_appcontext
def _teardown_cleanup(exception=None):
    try:
        gc.collect()
    except Exception:
        pass



# Ambil DB dari .env atau default ke SQLite
# engine, db_session = get_engine_and_session(DB_URL)


# Activate Data Science Page

# =========================
# Lazy-loaded resources
# =========================
_lock = threading.RLock()
_data = None
_products = [
    "Laptop", "Smartphone", "Tablet", "Headphone", "Smartwatch",
    "Kamera", "Speaker", "Mouse", "Keyboard"
]
_scaler = None
_kmeans = None
_churn_model = None
_nlp = None
_gemini_model = None
_gemini_configured = False

def get_products():
    return list(_products)

def _load_and_prepare_data():
    global _data, _scaler, _kmeans
    root_directory = os.getcwd()
    df = pd.read_csv(f"{root_directory}/data/customer_data.csv")
    # Preprocessing
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(df[["Age", "Income", "SpendingScore"]])
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    df["Cluster"] = kmeans.fit_predict(data_scaled)
    df["LoyaltyScore"] = (df["SpendingScore"] + (df["Income"] / 1000)) / 2
    df["CLV"] = df["SpendingScore"] * df["Income"] / 1000
    df["Churn"] = (df["SpendingScore"] < 30).astype(int)
    _data, _scaler, _kmeans = df, scaler, kmeans

def get_data():
    global _data
    with _lock:
        if _data is None:
            _load_and_prepare_data()
        return _data

def get_scaler():
    global _scaler
    with _lock:
        if _scaler is None:
            _load_and_prepare_data()
        return _scaler

def get_kmeans():
    global _kmeans
    with _lock:
        if _kmeans is None:
            _load_and_prepare_data()
        return _kmeans

def get_churn_model():
    global _churn_model
    with _lock:
        if _churn_model is None:
            df = get_data()
            X = df[["Age", "Income", "SpendingScore", "LoyaltyScore", "CLV"]]
            y = df["Churn"]
            model = xgb.XGBClassifier(objective="binary:logistic", base_score=0.5)
            model.fit(X, y)
            _churn_model = model
        return _churn_model


#  INFORMATION EXTRACTOR (Lazy)
# Load model NER dari Hugging Face saat pertama kali dipakai
def get_nlp():
    global _nlp
    with _lock:
        if _nlp is None:
            model_name = "dslim/distilbert-NER"
            # tokenizer = AutoTokenizer.from_pretrained(model_name)
            _nlp = pipeline("ner", model=model_name)
        return _nlp


# Chatbot Init Gemini Model (Lazy)
def get_gemini_model():
    global _gemini_model, _gemini_configured
    with _lock:
        if not _gemini_configured:
            genai.configure(api_key=GEMINI_API_KEY)
            _gemini_configured = True
        if _gemini_model is None:
            _gemini_model = genai.GenerativeModel(CHATBOT_GEMINI_MODEL)
        return _gemini_model
