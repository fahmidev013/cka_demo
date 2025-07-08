from flask import Flask, request

app = Flask(__name__)

@app.route("/")
def welcome():
    return "<p>App is running</p>"