from packages import *
from utils.contants import BASE_URL

def informationExtractorPage():
    st.title("PDF Information Extractor")
    st.write("Information Extractor digunakan untuk mengekstrak teks dan entitas dari file PDF. Anda dapat mengunggah file PDF dan melihat teks yang diekstrak beserta entitas yang teridentifikasi.")

    uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

    if uploaded_file:
        files = {"file": uploaded_file.getvalue()}
        
        with st.spinner("Processing..."):
            response = requests.post(f'{BASE_URL}/extract', files=files)
            
            if response.status_code == 200:
                data = response.json()
                st.subheader("Extracted Text:")
                st.text_area("Text", data["text"], height=300)

                st.subheader("Extracted Entities:")
                for entity in data["entities"]:
                    st.write(f'{entity} : {data["entities"][entity]}')
            else:
                st.error("Failed to process the file.")