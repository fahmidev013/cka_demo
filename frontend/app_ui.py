from packages import st

# Sidebar Navigasi
st.sidebar.title("CKA Artificial Intelligence")
menu = st.sidebar.radio("List Application", ["Chatbot", "Web Scrapping", "Information Extractor", "Data Science", "Face Recognition"])

# -------------------------
# 4. Info Monitoring (import psutil only when needed)
# -------------------------
try:
    import psutil, os
    process = psutil.Process(os.getpid())
    mem_usage = process.memory_info().rss / (1024 ** 2)
    st.sidebar.write(f"**Penggunaan Memori:** {mem_usage:.2f} MB")
except Exception:
    pass

# Routing halaman berdasarkan menu yang dipilih (lazy import per page)
if menu == "Chatbot":
    from page.chatbotPage import chatbotPage
    chatbotPage()
elif menu == "Data Science":
    from page.dataSciencePage import dataSciencePage
    dataSciencePage()
elif menu == "Information Extractor":
    from page.informationExtractorPage import informationExtractorPage
    informationExtractorPage()
# elif menu == "Face Recognition":
#     from frontend.page.faceRecognitionPage import faceRecognitionPage
#     faceRecognitionPage()
elif menu == "Web Scrapping":
    from page.webScrappingPage import webScrappingPage
    webScrappingPage()