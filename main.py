from backend import *
from frontend.utils import *
import random
 

@app.route('/chatbot', methods=['POST'])
def chatGeminiModel():
    data = request.json
    user_message = data.get('message')

    if not user_message:
        return jsonify({"error": "No message provided"}), 400
    print(user_message)
    try:
        # Panggil Gemini API
        response = model.generate_content(user_message)
        bot_response = response.text
        return jsonify({"response": bot_response})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/search', methods=['GET'])
def search_nearby_places():
    lat = request.args.get('lat', '6.874138255514192')  # format: "lat,lng"
    lng = request.args.get('lng', '107.50300964106383')  # format: "lat,lng"
    radius = request.args.get('radius', 5000)  # dalam meter
    keyword = request.args.get('keyword', 'perusahaan')
    url = GOOGLE_PLACES_URL
    headers = {
        "Content-Type": "application/json",
        "X-Goog-Api-Key": GOOGLE_API_KEY,
        "X-Goog-FieldMask": "places.displayName.text,places.formattedAddress,places.types,places.internationalPhoneNumber,places.rating,places.userRatingCount,places.websiteUri,places.reviews.originalText.text,places.postalAddress.locality"
        # "X-Goog-FieldMask": "*"
    }
    body = {
        "textQuery": keyword,
        "languageCode": "id",
        "regionCode": "id",
        "maxResultCount": 6,
        "locationBias": {
            "circle": {
                "center": {
                    "latitude": float(lat),
                    "longitude": float(lng)
                },
                "radius": float(radius)
            }
        },
    }
    
    try:
        response = requests.post(url, headers=headers, json=body)
        data = response.json()
    except Exception as e:
        print(f"KEsalahan terjadi adalah : {e}")
    results = []
    for place in data.get("places", []):
        name = place.get("displayName").get("text")
        address = place.get("formattedAddress")
        phone = place.get("internationalPhoneNumber")
        web = place.get("websiteUri")
        rating = place.get("rating")
        userRatingCount = place.get("userRatingCount")
        reviews = []
        if place.get("reviews"):
            for feedback in place.get("reviews"):
                reviews.append(feedback.get("originalText", {}).get("text"))
        types = place.get("types")
        locality = place.get("postalAddress", {}).get("locality")
        profile_info = [] # scrape_data(name)

        results.append({
            "name": name,
            "address": address,
            "phone": phone,
            "web": web,
            "rating": rating,
            "userRatingCount": userRatingCount,
            "reviews": reviews,
            "types": types,
            "locality": locality,
            "profile_info": profile_info
        })
        
    #     if len(profile_info) > 0:
    #         company = Company(
    #             name=name,
    #             address=address,
    #             phone=phone,
    #             web=str(web) if web is not None else '',
    #             rating=rating,
    #             user_ratings_count=userRatingCount,
    #             reviews=", ".join(str(x) if x is not None else '' for x in reviews),
    #             types=", ".join(str(x) if x is not None else '' for x in types),
    #             profile_info=", ".join(str(x) if x is not None else '' for x in profile_info)
    #         )
            
    #         db_session.add(company)
    # db_session.commit()
        
        
    return jsonify(results)


@app.route("/extract", methods=["POST"])
def extract_information():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    root_directory = os.getcwd()
    random_integer = random.randint(1, 99)
    file = request.files["file"]
    file_path = f"{root_directory}\\data\\upload\\demofile_{random_integer}.pdf"
    file.save(file_path)

    # Ekstraksi teks
    text = extract_text_from_pdf(file_path)

    # Ekstraksi entitas menggunakan model NLP
    entities = nlp(text)
    categorized_entities = categorize_entities(entities)
    return jsonify({"text": text, "entities": categorized_entities})



@app.route('/customers', methods=['GET'])
def get_customers():
    return jsonify(data.to_dict(orient="records"))


# Endpoiunt untuk Prediksi Cluster
@app.route('/predict', methods=['POST'])
def predict():
    try:
        request_data = request.get_json()
        age = request_data["Age"]
        income = request_data["Income"]
        spending = request_data["SpendingScore"]

        # Standarisasi input
        input_scaled = scaler.transform([[age, income, spending]])
        cluster = kmeans.predict(input_scaled)

        return jsonify({"Cluster": int(cluster[0])})
    except Exception as e:
        return jsonify({"error": str(e)})

# Download LAPORAN ANALISA PELANGGAN
@app.route('/report', methods=['GET'])
def generate_report():
    pdf_filename = "customer_report.pdf"
    c = canvas.Canvas(pdf_filename, pagesize=letter)
    c.drawString(100, 750, "Customer Analytics Report")
    
    y_position = 730
    for index, row in data.iterrows():
        text = f"{row['Name']} - Age: {row['Age']}, Income: {row['Income']}, Spending Score: {row['SpendingScore']}, Cluster: {row['Cluster']}"
        c.drawString(100, y_position, text)
        y_position -= 20
        if y_position < 100:
            c.showPage()
            y_position = 750

    c.save()
    return send_file(pdf_filename, as_attachment=True)

# Endpoint untuk mengukur tingkat resiko pelanggan
@app.route('/report_churn', methods=['GET'])
def generate_report_churn():
    pdf_filename = "customer_report_churn.pdf"
    c = canvas.Canvas(pdf_filename, pagesize=letter)
    c.drawString(100, 750, "Customer Analytics Report")
    
    y_position = 730
    for index, row in data.iterrows():
        text = f"{row['Name']} - Loyalty: {row['LoyaltyScore']:.2f}, CLV: {row['CLV']:.2f}, Churn Risk: {row['Churn']}"
        c.drawString(100, y_position, text)
        y_position -= 20
        if y_position < 100:
            c.showPage()
            y_position = 750

    c.save()
    return send_file(pdf_filename, as_attachment=True)

# **Fungsi Analisis Sentimen**
def analyze_sentiment(text):
    sentiment = TextBlob(text).sentiment.polarity
    return "Positif" if sentiment > 0 else "Negatif" if sentiment < 0 else "Netral"

data["Sentiment"] = data["Review"].apply(analyze_sentiment)

@app.route('/reviews', methods=['GET'])
def get_reviews():
    return jsonify(data[["Name", "Review", "Sentiment"]].to_dict(orient="records"))


# Enpoint untuk Sentimen Ulasan
@app.route('/analyze_review', methods=['POST'])
def analyze_review():
    try:
        request_data = request.get_json()
        review_text = request_data["Review"]
        sentiment = analyze_sentiment(review_text)
        return jsonify({"Review": review_text, "Sentiment": sentiment})
    except Exception as e:
        return jsonify({"error": str(e)})


# **Fungsi Rekomendasi Produk**
def recommend_products(customer_name):
    if customer_name not in data["Name"].values:
        return []
    
    # Ambil data pelanggan
    customer_index = data[data["Name"] == customer_name].index[0]
    # Hitung kemiripan dengan pelanggan lain
    product_matrix = data.iloc[:, 1:].values
    similarities = cosine_similarity(product_matrix)
    
    
    # Rekomendasi berdasarkan pelanggan paling mirip
    similar_customer_index = np.argsort(similarities[customer_index])[-2]  # Ambil pelanggan paling mirip selain dirinya sendiri
    recommended_products = []
    
    for product in products:
        if data.loc[customer_index, product] == 0 and data.loc[similar_customer_index, product] == 1:
            recommended_products.append(product)
    
    return recommended_products

#Endpoint untuk Sistem Rekomendasi
@app.route('/recommend', methods=['POST'])
def get_recommendations():
    try:
        request_data = request.get_json()
        customer_name = request_data["Name"]
        recommendations = recommend_products(customer_name)
        print(recommendations)
        return jsonify({"Name": customer_name, "Recommendations": str(recommendations)})
    except Exception as e:
        return jsonify({"error": str(e)})


@app.route('/predict_churn', methods=['POST'])
def predict_churn():
    try:
        request_data = request.get_json()
        age_churn = request_data["Age"]
        income_churn = request_data["Income"]
        spending_churn = request_data["SpendingScore"]
        
        loyalty = (spending_churn + (income_churn / 1000)) / 2
        clv = spending_churn * income_churn / 1000

        input_data = np.array([[age_churn, income_churn, spending_churn, loyalty, clv]])
        churn_prob = churn_model.predict_proba(input_data)[0][1]  
        return jsonify({"Churn_Probability": float(churn_prob)})
    except Exception as e:
        print('Res Error')
        return jsonify({"error": str(e)})