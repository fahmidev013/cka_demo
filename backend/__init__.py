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
import pandas as pd
import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import time
from sqlalchemy import Column, Integer, Float, String, Text, create_engine
from sqlalchemy.orm import declarative_base, sessionmaker
from data.models import Company, get_engine_and_session

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
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
customers_data = pd.read_csv(f"{root_directory}\\data\\customer_data.csv")
# Load dataset
data = pd.read_csv(f"{root_directory}\\data\\customer_data.csv")


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


from transformers import AutoTokenizer, TFAutoModelForTokenClassification
import tensorflow as tf
import streamlit as st

# Load model dan tokenizer
MODEL_NAME = "aadhistii/DistilBERT-Indonesian-NER"

@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = TFAutoModelForTokenClassification.from_pretrained(MODEL_NAME)
    return tokenizer, model

tokenizer, model = load_model()

def extract_ner(text):
    inputs = tokenizer(text.split(),
                       is_split_into_words=True,
                       return_tensors="tf",
                       padding=True,
                       truncation=True)

    outputs = model(inputs)
    logits = outputs.logits
    predictions = tf.argmax(logits, axis=-1).numpy()[0]
    tokens = inputs["input_ids"][0].numpy()
    print(f"token  {tokens}")
    print(f"predictions  {predictions}")

    labels = model.config.id2label
    print(f"labels  {labels}")
    results = []

    for token_id, pred_id in zip(tokens, predictions):
        token = tokenizer.decode([token_id])
        label = labels[pred_id]
        results.append((token, label))
        print(f"HASIL  {token} -- {label}")
        # if label != "O":
        #     results.append((token, label))
    print(f"results  {results}")
    return results


# from transformers import AutoModel, AutoTokenizer, pipeline
# model_name = "aadhistii/DistilBERT-Indonesian-NER"

# model = AutoModel.from_pretrained(model_name, from_tf=True)
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# nlp = pipeline(tokenizer, model=f"{model}")

if __name__ == '__main__':
    app.run()

