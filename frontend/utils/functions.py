from backend import *

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
