from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from bs4 import BeautifulSoup
import time

# # Set up Chrome options
# chrome_options = Options()
# chrome_options.add_argument("--headless")  # Run in headless mode
# chrome_options.add_argument("--disable-gpu")
# chrome_options.add_argument("--no-sandbox")

# # Set up the Chrome driver
# service = Service('/usr/local/bin/chromedriver')  # Adjust the path if necessary
# driver = webdriver.Chrome(service=service, options=chrome_options)

# # Navigate to the URL
# driver.get('https://www.thewhiskyexchange.com/c/35/japanese-whisky')

# # Get the page source and parse it with BeautifulSoup

# # Here I need to get all the item's links then to access to each item info, but this didn't work
# soup = BeautifulSoup(driver.page_source, 'html.parser')
# productlist = soup.find_all('li', class_='product-grid__item')

# # Print the product list
# print(productlist)

# # Close the driver
# driver.quit()

# Binh
# With head mode
# url = "https://www.thewhiskyexchange.com/c/35/japanese-whisky"

# driver = webdriver.Chrome()

# driver.get(url)
# html = driver.page_source
# soup = BeautifulSoup(html, 'lxml')

# products = soup.find_all('li', class_="product-grid__item")
# print(products)
# driver.quit()

# Headless mode
# url = "https://www.thewhiskyexchange.com/c/35/japanese-whisky"

# # Set up Chrome options for headless mode
# chrome_options = Options()
# chrome_options.add_argument("--headless")      # Run in headless mode
# chrome_options.add_argument("--disable-gpu")   # Recommended for headless
# chrome_options.add_argument("--no-sandbox")    # Recommended for some environments

# # Initialize the driver with headless options
# driver = webdriver.Chrome(options=chrome_options)

# try:
#     # Navigate to the URL
#     driver.get(url)
    
#     # Get the page source
#     html = driver.page_source
    
#     # Parse with BeautifulSoup
#     soup = BeautifulSoup(html, 'lxml')
    
#     # Find products
#     products = soup.find_all('li', class_="product-grid__item")
#     print(products)

# finally:
#     # Make sure to quit the driver at the end
#     driver.quit()

# Headless mode Amazon link test
url = "https://www.amazon.com/Best-Sellers-Clothing-Shoes-Jewelry/zgbs/fashion/ref=zg_bs_nav_fashion_0"

# Set up Chrome options for headless mode
chrome_options = Options()
chrome_options.add_argument("--headless")      # Run in headless mode
chrome_options.add_argument("--disable-gpu")   # Recommended for headless
chrome_options.add_argument("--no-sandbox")    # Recommended for some environments

# Initialize the driver with headless options
driver = webdriver.Chrome(options=chrome_options)

try:
    # Navigate to the URL
    driver.get(url)
    
    # Get the page source
    html = driver.page_source
    
    # Parse with BeautifulSoup
    soup = BeautifulSoup(html, 'lxml')
    
    # Find products
    products = soup.find_all('div', class_="zg-grid-general-faceout")
    print(products)

finally:
    # Make sure to quit the driver at the end
    driver.quit()

# Test with bot bypassing
# url = "https://www.thewhiskyexchange.com/c/35/japanese-whisky"

# # 1) Configure Chrome to run headless.
# chrome_options = Options()
# chrome_options.add_argument("--headless")  # headless mode
# chrome_options.add_argument("--disable-gpu")
# chrome_options.add_argument("--no-sandbox")
# chrome_options.add_argument("--disable-blink-features=AutomationControlled")
# chrome_options.add_argument("--disable-dev-shm-usage")

# # 2) Create the driver.
# driver = webdriver.Chrome(options=chrome_options)

# # 3) Inject a small snippet of JS to disguise webdriver in the page context.
# #    (Sometimes used to bypass basic bot checks that look for navigator.webdriver.)
# driver.execute_cdp_cmd(
#     "Page.addScriptToEvaluateOnNewDocument",
#     {
#         "source": """
#           Object.defineProperty(navigator, 'webdriver', {
#             get: () => undefined
#           });
#         """
#     },
# )

# try:
#     # 4) Visit the page.
#     driver.get(url)

#     # 5) Wait a moment to let the page (and any JS) load fully.
#     #    You might need to increase this if the site is very slow or uses JS to load items.
#     time.sleep(5)

#     # 6) Optionally scroll to the bottom of the page to ensure lazy-loaded elements appear.
#     driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
#     time.sleep(3)

#     # 7) Now grab the page source, parse with BeautifulSoup.
#     html = driver.page_source
#     soup = BeautifulSoup(html, "lxml")

#     # 8) Scrape the desired items.
#     products = soup.find_all('li', class_="product-grid__item")
#     print(f"Found {len(products)} product items.")
#     for p in products:
#         print(p.get_text(strip=True))

# finally:
#     # 9) Quit the driver.
#     driver.quit()