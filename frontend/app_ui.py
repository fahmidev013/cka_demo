from packages import *

# Sidebar Navigasi
st.sidebar.title("CKA Artificial Intelligence")
menu = st.sidebar.radio("List Application", ["Chatbot", "Web Scrapping", "Data Science", "Information Extractor", "Face Recognition"])

# Routing halaman berdasarkan menu yang dipilih
if menu == "Chatbot":
    chatbotPage()
# elif menu == "Data Science":
#     dataSciencePage()
# elif menu == "Information Extractor":
#     informationExtractorPage()
# elif menu == "Face Recognition":
#     faceRecognitionPage()
elif menu == "Web Scrapping":
    webScrappingPage()