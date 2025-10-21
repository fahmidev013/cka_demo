import pdfplumber

# Ekstraksi teks dari PDF
def extract_text_from_pdf(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text() + "\n"
    return text

# Kategorisasi entitas
def categorize_entities(entities):
    categorized = {"PERSON": [], "ORG": [], "LOC": [], "CONTACT": []}
    for entity in entities:
        label = entity['entity']
        if label == "B-PER" or label == "I-PER":
            categorized["PERSON"].append(entity['word'])
        elif label == "B-ORG" or label == "I-ORG":
            categorized["ORG"].append(entity['word'])
        elif label == "B-LOC" or label == "I-LOC":
            categorized["LOC"].append(entity['word'])
        # Identifikasi kontak dari teks secara manual
        if "@" in entity['word'] or "+" in entity['word']:
            categorized["CONTACT"].append(entity['word'])
    return categorized
    

def scrape_data(company="ultrajaya"):
    options = Options()
    options.add_argument("--headless")  # Tanpa tampilan browser

    driver = webdriver.Chrome(options=options)
    driver.get(f"https://search.katadata.co.id/search?q={company}&source=databoks")

    # Tunggu konten termuat (delay JS)
    time.sleep(4)  # Bisa diganti dengan WebDriverWait
    
    
    # Ambil konten HTML setelah JS selesai render
    soup = BeautifulSoup(driver.page_source, 'html.parser')
    results = soup.find_all('p')
    
    companyInfo = []
    if len(results) > 0:
    # Lanjutkan parsing seperti biasa
        for item in results[:-2]:
            companyInfo.append(item.get_text(strip=True))

    driver.quit()

    return companyInfo

