from packages import *

# Sidebar Navigasi
st.sidebar.title("CKA Artificial Intelligence")
menu = st.sidebar.radio("List Application", ["Chatbot", "Web Scrapping", "Information Extractor", "Data Science", "Face Recognition"])

# -------------------------
# 4. Info Monitoring
# -------------------------
import psutil, os
process = psutil.Process(os.getpid())
mem_usage = process.memory_info().rss / (1024 ** 2)
st.sidebar.write(f"**Penggunaan Memori:** {mem_usage:.2f} MB")

# Routing halaman berdasarkan menu yang dipilih
if menu == "Chatbot":
    chatbotPage()
elif menu == "Data Science":
    dataSciencePage()
elif menu == "Information Extractor":
    informationExtractorPage()
# elif menu == "Face Recognition":
#     faceRecognitionPage()
elif menu == "Web Scrapping":
    webScrappingPage()