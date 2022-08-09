# Relevant Libraries
from bs4 import BeautifulSoup
import requests
import csv

# This is to scrape the whole website, using the libraries
hafizoglu_baklava_url = 'https://online.hafizmustafa.com/baklava-en'
source = requests.get(hafizoglu_baklava_url).text
soup = BeautifulSoup(source, 'lxml')

# These three lines are to scrape the data to a csv file
csv_file = open('baklava_scrape.csv', 'w')
csv_writer = csv.writer(csv_file)
csv_writer.writerow(['baklava_name', 'baklava_size', 'baklava_price','baklava_image_url']) # The headers


match = soup.find("div", {"id": "ProductPageProductList"})

base_url = 'https://online.hafizmustafa.com'
for product in match.find_all('div',class_ = 'ItemOrj col-lg-3 col-md-3 col-sm-6 col-xs-6'):

    full_name = product.find('div',class_= 'productName detailUrl')
    full_name = full_name.a.text

    try:
        full_name1 = full_name.split('(')
        baklava_name = full_name1[0]
        baklava_size = full_name1[1]
    except Exception as e:
        baklava_size = None
        baklava_name = full_name

    print(baklava_name)
    print(baklava_size)


    price = product.find('div', class_ = 'discountPrice').span.text
    print(price.strip())

    website_src = product.find('div', class_ = 'productImage').a.img['src']
    if website_src == '/Uploads/Images/load.gif':
        website_src = product.find('div', class_ = 'productImage').a.img['data-original']
    image_url = base_url + website_src
    print(image_url)
    print()
    csv_writer.writerow([baklava_name,baklava_size,price,image_url])

csv_file.close()
