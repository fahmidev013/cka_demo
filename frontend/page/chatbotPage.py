from packages import *
from utils.contants import BASE_URL

def chatbotPage():
    st.title(f"Chatbot")
    st.write("Chatbot adalah program komputer yang dirancang untuk meniru percakapan manusia dan memberikan respon yang mirip dengan manusia. Biasanya, chatbot digunakan untuk membantu pengguna dalam melakukan tugas-tugas tertentu atau memberikan informasi yang dibutuhkan. Chatbot menggunakan algoritma dan kecerdasan buatan untuk memahami bahasa manusia dan merespons dengan tepat. Seiring dengan perkembangan teknologi, chatbot semakin canggih dan mampu menangani percakapan yang lebih kompleks.")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    user_input = st.text_input("Anda:")

    if st.button("Kirim"):
        if user_input:
            response = requests.post(f'{BASE_URL}/chat', json={"message": user_input})
            bot_response = response.json().get("response", "Error")
            
            # Simpan percakapan ke dalam session state
            st.session_state.chat_history.append(("Anda", user_input))
            st.session_state.chat_history.append(("Chatbot", bot_response))

    # Tampilkan chat history dengan multi-scroll UI
    st.subheader("Percakapan")
    chat_container = st.container()

    with chat_container:
        for sender, message in st.session_state.chat_history:
            st.markdown(f"**{sender}:** {message}")
