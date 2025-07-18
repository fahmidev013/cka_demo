from packages import *
from utils.contants import BASE_URL



def informationExtractorPage():
    
    # Streamlit UI
    st.title("🧠 Information Extractor NER Bahasa Indonesia")
    text_input = st.text_area("Masukkan teks (contoh: Presiden Jokowi berkunjung ke Bandung hari Senin)", height=200)

    if st.button("Ekstrak Entitas"):
        if text_input.strip():
            # entities = extract_ner(text_input)
            response = requests.get(f'{BASE_URL}/extract')
            
            if response.status_code == 200:
                data = response.json()
                st.subheader("Extracted Text:")
                st.text_area("Text", data["text"], height=300)
                st.subheader("📌 Hasil Ekstraksi Entitas:")
                for token, label in data:
                    st.write(f"**{token}** → `{label}`")
            else:
                st.error("Failed to process the file.")
            
        else:
            st.warning("Teks tidak boleh kosong!")
    
    
    
    
    # st.title("PDF Information Extractor")
    # st.write("Information Extractor digunakan untuk mengekstrak teks dan entitas dari file PDF. Anda dapat mengunggah file PDF dan melihat teks yang diekstrak beserta entitas yang teridentifikasi.")

    # uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

    # if uploaded_file:
    #     files = {"file": uploaded_file.getvalue()}
        
    #     with st.spinner("Processing..."):
    #         response = requests.post(f'{BASE_URL}/extract', files=files)
            
    #         if response.status_code == 200:
    #             data = response.json()
    #             st.subheader("Extracted Text:")
    #             st.text_area("Text", data["text"], height=300)

    #             st.subheader("Extracted Entities:")
    #             for entity in data["entities"]:
    #                 st.write(f'{entity} : {data["entities"][entity]}')
    #         else:
    #             st.error("Failed to process the file.")