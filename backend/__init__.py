import os
from .functions import *
from flask import Flask, request, jsonify, send_file
from frontend.utils.contants import *
from frontend.utils.functions import *
from dotenv import load_dotenv
from flask_cors import CORS

import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import time
from sqlalchemy import Column, Integer, Float, String, Text, create_engine
from sqlalchemy.orm import declarative_base, sessionmaker
from data.models import Company, get_engine_and_session


# Data Science Libs
import polars as pl
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

import streamlit as st


# from transformers import pipeline

load_dotenv()

app = Flask(__name__)

# Untuk mengizinkan komunikasi dengan frontend
CORS(app)



# Ambil DB dari .env atau default ke SQLite
# engine, db_session = get_engine_and_session(DB_URL)


# Activate Data Science Page

# Load dataset
root_directory = os.getcwd()

@st.cache_resource
def load_customers_data():
    data = pl.scan_csv(f"{root_directory}/data/customer_data.csv")
    return data
customers_data = load_customers_data()
products = ["Laptop", "Smartphone", "Tablet", "Headphone", "Smartwatch", "Kamera", "Speaker", "Mouse", "Keyboard"]


# ===============================
# 🔹 Preprocessing: pilih kolom & materialize
# ===============================
# Ambil hanya kolom yang diperlukan, lalu collect → baru load ke memory
customers_data = customers_data.select(["Name","Review","Age", "Income", "SpendingScore"]).collect()

# Standarisasi data
scaler = StandardScaler()
data_scaled = scaler.fit_transform(
    customers_data.select(["Age", "Income", "SpendingScore"]).to_numpy()
)

# ===============================
# 🔹 K-Means Clustering
# ===============================
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
clusters = kmeans.fit_predict(data_scaled)

customers_data = customers_data.with_columns(
    pl.Series("Cluster", clusters)
)

# ===============================
# 🔹 Loyalty Score
# ===============================
customers_data = customers_data.with_columns(
    ((pl.col("SpendingScore") + (pl.col("Income") / 1000)) / 2)
    .alias("LoyaltyScore")
)

# ===============================
# 🔹 Customer Lifetime Value (CLV)
# ===============================
customers_data = customers_data.with_columns(
    (pl.col("SpendingScore") * pl.col("Income") / 1000)
    .alias("CLV")
)

# ===============================
# 🔹 Prediksi Churn
# SpendingScore < 30 → churn
# ===============================
customers_data = customers_data.with_columns(
    (pl.col("SpendingScore") < 30).cast(pl.Int64).alias("Churn")
)

# ===============================
# 🔹 Model Prediksi Churn (XGBoost)
# ===============================
X = customers_data.select(["Age", "Income", "SpendingScore", "LoyaltyScore", "CLV"]).to_numpy()
y = customers_data["Churn"].to_numpy()

churn_model = xgb.XGBClassifier(
    objective="binary:logistic",
    base_score=0.5,
    max_bin=256,
    tree_method="hist",
    max_depth=4
)
churn_model.fit(X, y)




#  INFORMATION XTRACTOR
# Load model NER dari Hugging Face
# nlp = pipeline("ner", model="dbmdz/bert-base-cased-finetuned-conll03-english")
# nlp = pipeline("ner", model="dslim/distilbert-NER")
# nlp = pipeline("ner", from_tf=True, model="aadhistii/DistilBERT-Indonesian-NER")
# nlp = pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english")
# model_name = "gagan3012/bert-tiny-finetuned-ner"

from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification


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
model = genai.GenerativeModel(CHATBOT_GEMINI_MODEL)


# from transformers import AutoModel, AutoTokenizer, pipeline
# model_name = "aadhistii/DistilBERT-Indonesian-NER"

# model = AutoModel.from_pretrained(model_name, from_tf=True)
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# nlp = pipeline(tokenizer, model=f"{model}")

if __name__ == '__main__':
    app.run()

