from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from bs4 import BeautifulSoup
import requests
import random
import time

# To generate the right HEADERS dictionary for your web scraping use case, follow these steps:

# 1. Open Developer Tools in your web browser: Go to the website you want to scrape and open the Developer Tools. Usually this can be done by right-clicking on the page and selecting "Inspect" or "Inspect Element".
# 2. Go to the Network Tab: In the Developer Tools, find and click on the "Network" tab.
# 3. Reload the page: Refresh the website while the Network tab is open. You will see a list of network requests being made.
# 4. Find the main request: Look for the main request that loads the HTML content of the page. This is usually the first or one of the top requests in the list.
# 5. Inspect the Headers: Click on the main request to see its details. Find the "Headers" tab within the request details.
# 6. Copy the User-Agent and Accept-Language: Look for the "User-Agent" and "Accept-Language" headers and copy their values.
# 7. Create the HEADERS dictionary: In your Python code, create a dictionary like the one in your example and paste the copied header values.

# This lots of user agents are for fooling Amazon that many requests are not from the same user
useragents=['Mozilla/5.0 (Macintosh; Intel Mac OS X 10_12_2) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/100.0.4896.88 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/100.0.4896.127 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_14_2) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/100.0.4896.127 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 11_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/100.0.4894.117 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 11_4) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/100.0.4855.118 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_4) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/100.0.4896.127 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_12_0) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/100.0.4896.127 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_14_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/100.0.4896.88 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/100.0.4892.86 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 11_4) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/100.0.4854.191 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 11_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/100.0.4859.153 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_14_3) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/100.0.4896.79 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/100.0.4896.127 Safari/537.36/null',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/100.0.4896.127 Safari/537.36,gzip(gfe)',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_4) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/100.0.4895.86 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 12_3_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/100.0.4896.127 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 11_13) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/100.0.4860.89 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 11_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/100.0.4885.173 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/100.0.4864.0 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 11_12) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/100.0.4877.207 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_12_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/100.0.4896.127 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 12_2_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/100.0.4896.60 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 11_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/100.0.4896.127 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_12_6) AppleWebKit/537.36 (KHTML%2C like Gecko) Chrome/100.0.4896.127 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_14_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/100.0.4896.133 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_16_0) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/100.0.4896.75 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_12) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/100.0.4872.118 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 12_3_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/100.0.4896.88 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 11_13) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/100.0.4876.128 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_14_3) AppleWebKit/537.36 (KHTML%2C like Gecko) Chrome/100.0.4896.127 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_3) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/100.0.4896.127 Safari/537.36']

headers={
    "User-Agent":useragents[random.randint(0,31)],
    "Accept-Language": "en-US,en;q=0.9",
}

# Amazon Best Sellers URL
url = "https://www.amazon.com/Best-Sellers-Clothing-Shoes-Jewelry/zgbs/fashion/ref=zg_bs_nav_fashion_0"

def get_all_items(URL, HEADERS):
    # Send a GET request
    response = requests.get(URL, headers=HEADERS)

    # Find all item links on the page
    links = []
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, "html.parser")
        
        for a_tag in soup.find_all("div", class_="zg-grid-general-faceout"):
            # print(f"a_tag: {a_tag}")
            href = a_tag.find("a")["href"]
            # print(f"href: {href}")
            full_link = "https://www.amazon.com" + href
            links.append(full_link)

        # Remove duplicates
        unique_links = list(set(links))

        # Print the links
        print("Found", len(unique_links), "links:")
        # for link in unique_links:
        #     print(link)
        return unique_links
    else:
        print("Failed to fetch the page. Status code:", response.status_code)
        return links
    
def display_and_save_result(URL, HEADERS):
    # opening our output file in append mode
    File = open("out.csv", "a")

    # Making the HTTP Request
    webpage = requests.get(URL, headers=HEADERS)

    # Creating the Soup Object containing all data
    soup = BeautifulSoup(webpage.content, "lxml")

    # retrieving product title
    try:
        # Outer Tag Object
        title = soup.find("span", attrs={
            'id': 'productTitle'
        })

        # Inner NavigableString Object
        title_value = title.string

        # Title as a string value
        title_string = title_value.strip().replace(',', '')

    except AttributeError:
        title_string = "NA"
    print("product Title = ", title_string)

    # saving the title in the file
    File.write(f"{title_string},")

    # Binh's comment:
    # This id priceblock might not be practical since I always see NA whichever links I scrap.
    # And seems like the price on Amazon will always change. So a price tracker could be a good approach for this
    # Please refer to the link to understand how to implement a price tracker and integrate it in this code
    # https://www.youtube.com/watch?v=qukjS96clB8
    
    # retrieving price
    try:
        price = soup.find("span", attrs={
            'id': 'priceblock_ourprice'
        }).string.strip().replace(',', '')
        # we are omitting unnecessary spaces
        # and commas form our string
    except AttributeError:
        price = "NA"
    print("Products price = ", price)

    # saving
    File.write(f"{price},")

    # retrieving product rating
    try:
        rating = soup.find("i", attrs={
                           'class': 'a-icon a-icon-star a-star-4-5'
        }).string.strip().replace(',', '')

    except AttributeError:

        try:
            rating = soup.find("span", attrs={
                'class': 'a-icon-alt'
            }).string.strip().replace(',', '')
        except:
            rating = "NA"
    print("Overall rating = ", rating)

    File.write(f"{rating},")

    try:
        review_count = soup.find("span", attrs={
            'id': 'acrCustomerReviewText'
        }).string.strip().replace(',', '')

    except AttributeError:
        review_count = "NA"
    print("Total reviews = ", review_count)
    File.write(f"{review_count},")

    # Binh's comment: This one might also need a tracker too
    # print availablility status
    try:
        available = soup.find("div", attrs={'id': 'availability'})
        available = available.find("span").string.strip().replace(',', '')

    except AttributeError:
        available = "NA"
    print("Availability = ", available)

    # saving the availability and closing the line
    File.write(f"{available},\n")

    ####################################################################################################################
    ### CAN WE ALSO GET OTHER INFO?                                                                                  ###
    ### COLOR, SIZE, PRODUCT DETAILS (FABRIC, CARE INSTRUCTIONS, ORIGIN, ETC), ABOUT THIS ITEM (PRODUCT DESCRIPTION) ###
    ####################################################################################################################

    # closing the file
    File.close()

def main():
    links = get_all_items(url, headers)
    for link in links:
        display_and_save_result(link, headers)

if __name__ == "__main__":
    main()



