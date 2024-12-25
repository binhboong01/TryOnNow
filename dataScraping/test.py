from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from bs4 import BeautifulSoup

# Set up Chrome options
chrome_options = Options()
chrome_options.add_argument("--headless")  # Run in headless mode
chrome_options.add_argument("--disable-gpu")
chrome_options.add_argument("--no-sandbox")

# Set up the Chrome driver
service = Service('/usr/local/bin/chromedriver')  # Adjust the path if necessary
driver = webdriver.Chrome(service=service, options=chrome_options)

# Navigate to the URL
driver.get('https://www.thewhiskyexchange.com/c/35/japanese-whisky')

# Get the page source and parse it with BeautifulSoup

# Here I need to get all the item's links then to access to each item info, but this didn't work
soup = BeautifulSoup(driver.page_source, 'html.parser')
productlist = soup.find_all('li', class_='product-grid__item')

# Print the product list
print(productlist)

# Close the driver
driver.quit()