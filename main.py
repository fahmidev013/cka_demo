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
        
        if len(profile_info) > 0:
            company = Company(
                name=name,
                address=address,
                phone=phone,
                web=str(web) if web is not None else '',
                rating=rating,
                user_ratings_count=userRatingCount,
                reviews=", ".join(str(x) if x is not None else '' for x in reviews),
                types=", ".join(str(x) if x is not None else '' for x in types),
                profile_info=", ".join(str(x) if x is not None else '' for x in profile_info)
            )
            
            db_session.add(company)
    db_session.commit()
        
        
    return jsonify(results)