from backend import *
from frontend.utils import *
import random


@app.route('/customers', methods=['GET'])
def get_customers():
    return jsonify(customers_data.to_dict(orient="records"))

@app.route('/reviews', methods=['GET'])
def get_reviews():
    return jsonify(data[["Name", "Review", "Sentiment"]].to_dict(orient="records"))



@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    user_input = data.get("message", "")
    openaikey = OPENAI_API_KEY
    
    
    if not user_input:
        return jsonify({"error": "No input provided"}), 400
    
    response = conversation.predict(input=user_input)
    return jsonify({"response": response})

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


@app.route("/extract", methods=["GET"])
def extract_information():
    # if "file" not in request.files:
    #     return jsonify({"error": "No file uploaded"}), 400

    # root_directory = os.getcwd()
    # random_integer = random.randint(1, 99)
    # file = request.files["file"]
    # file_path = f"{root_directory}\\data\\upload\\demofile_{random_integer}.pdf"
    # file.save(file_path)

    # Ekstraksi teks
    # text = extract_text_from_pdf(file_path)
    
    text = "Presiden Joko Widodo mengunjungi Kota Bandung pada hari Senin."
    ner_results = extract_ner(text)
    result = {}
    for token, label in ner_results:
        result[token] = label
        # print(f"{token} => {label}")


    # Ekstraksi entitas menggunakan model NLP
    # entities = nlp(text)
    # categorized_entities = categorize_entities(entities)
    # print(categorized_entities)
    print(ner_results)
    print(f" RESULT  {result}")
    return jsonify({"text": ner_results, "entities": result})


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