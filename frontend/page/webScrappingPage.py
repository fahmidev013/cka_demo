from packages import *
from utils.contants import BASE_URL
from streamlit_geolocation import streamlit_geolocation
import streamlit.components.v1 as components
import pandas as pd

def webScrappingPage():
    # st.set_page_config(page_title="Pencarian Perusahaan Otomatis", layout="centered")
    st.title("🔍 Cari Perusahaan di Sekitar Saya")
    location = streamlit_geolocation()
    # Get location from URL query params
    auto_loc = f"{location['latitude']}, {location['longitude']}" 

    # --- Form input ---
    with st.form("search_form"):
        if auto_loc:
            lat_default, lng_default = map(float, auto_loc.split(","))
            st.success(f"📍 Lokasi Terdeteksi: {lat_default}, {lng_default}")
        else:
            lat_default, lng_default = -6.87, 107.51
            st.info("📍 Menunggu lokasi... atau masukkan manual")

        col1, col2 = st.columns(2)
        with col1:
            lat = st.number_input("Latitude", value=lat_default)
        with col2:
            lng = st.number_input("Longitude", value=lng_default)

        radius = st.slider("Radius Pencarian (meter)", 100, 50000, 10000, step=50)
        keyword = st.text_input("Keyword", value="perusahaan")
        submitted = st.form_submit_button("🔍 Cari Perusahaan Sekitar")
        
    # url = f"http://localhost:5000/api/search_with_scrape?location={f"{lat},{lng}"}&radius={radius}"
    if submitted:
        with st.spinner("🔄 Mencari perusahaan..."):
            params = {
                "lat": lat,
                "lng": lng,
                "radius": radius,
                "keyword": keyword
            }

            try:
                res = requests.get(f"{BASE_URL}/api/search", params=params)
                res.raise_for_status()
                data = res.json()

                if not data:
                    st.warning("Tidak ditemukan perusahaan.")
                else:
                    df = pd.DataFrame(data)
                    st.success(f"Ditemukan {len(df)} perusahaan.")
                    st.dataframe(df)

                    if "latitude" in df.columns and "longitude" in df.columns:
                        st.map(df.rename(columns={"latitude": "lat", "longitude": "lon"}))

                    st.download_button(
                        "⬇️ Download CSV",
                        data=df.to_csv(index=False).encode("utf-8"),
                        file_name="data_perusahaan.csv",
                        mime="text/csv"
                    )
                    count = 1
                    for d in data:
                        st.write(f"## 🏢 {count} **{d['name']}**")
                        st.write(f"##### ALAMAT: {d.get('address', '-')}")
                        st.write(f"##### RATING: {d.get('rating', '-')}")
                        st.write(f"##### JUMLAH USER RATING: {d.get('userRatingCount', '-')}")
                        st.write(f"##### PHONE: {d.get('phone', '-')} — (Web: {d.get('web', '-')})")
                        st.write(f"##### JENIS: {", ".join(d.get('types', '-'))}")
                        st.write(f"___ :red[INFO]:___ {", ".join(d.get('profile_info', '-'))}")
                        if d.get('reviews', '-'):
                            for feedback in d.get('reviews', '-'):
                                st.write(f"______ Feedback :_____ {feedback}")
                        count = count + 1
                    


            except Exception as e:
                st.error(f"❌ Error saat request: {e}")
