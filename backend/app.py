from backend import *
from frontend.utils import *


# Load dataset
root_directory = os.getcwd()
customers_data = pd.read_csv(f"{root_directory}/data/customer_data.csv")

@app.route('/customers', methods=['GET'])
def get_customers():
    return jsonify(customers_data.to_dict(orient="records"))

@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    user_input = data.get("message", "")
    openaikey = OPENAI_API_KEY
    
    
    if not user_input:
        return jsonify({"error": "No input provided"}), 400
    
    response = conversation.predict(input=user_input)
    return jsonify({"response": response})

