from backend import *
from frontend.utils import *

@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    user_input = data.get("message", "")
    openaikey = OPENAI_API_KEY
    
    
    if not user_input:
        return jsonify({"error": "No input provided"}), 400
    
    response = conversation.predict(input=user_input)
    return jsonify({"response": response})

