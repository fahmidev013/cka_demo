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

