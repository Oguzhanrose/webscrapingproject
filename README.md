`Python`, `Webscraping`,`Pandas`, `Plotly`, `Data Cleaning`, `Data Visualizations`, `Basic Statistics`, `Interactive plots` 

# Webscraping project - With data cleaning and visualization

This personal project is made to understand and learn how data can be scraped from a website. Baklava is a very delicious turkish dessert, and I really like 
baklava. I therefore decided to webscrape [Hafız Mustafa](https://online.hafizmustafa.com/baklava-en), which is a baklava brand, which in my opinion makes 
the best baklava. The following three images are respectively a webscraped baklava image from the website, the sub-dataset (the whole dataset contain 64 rows) 
I have created by scraping data from the website and cleaning it with Pandas and an interactive visualization made with Plotly.

<br>
<p align="center"> <img src="./imagefolder/fistikli-cevizli-karisik-baklava-xl-ku-436-99.jpg" alt="Drawing"/> </p>
<br>

<br>
<p align="center"> <img src="./git_image/webscraped_and_cleaned_dataframe.png" alt="Drawing"/> </p>
<br>

<br>
<p align="center"> <img src="./git_image/plotly_visualization.png" alt="Drawing"/> </p>
<br>

## 1.) Webscraping
So we start first with scraping the data from this [website](https://online.hafizmustafa.com/baklava-en).

### 1.1) Import relevant libraries

<br>

<details>
<summary>Click to see the libraries</summary>

```python
# Relevant Libraries 
from bs4 import BeautifulSoup
import requests
import csv
```

</details>

<br>

### 1.2) Fetch content from website
```python
# By this code we fetch the content from the URL given
hafizoglu_baklava_url = 'https://online.hafizmustafa.com/baklava-en'
source = requests.get(hafizoglu_baklava_url).text
soup = BeautifulSoup(source, 'lxml')
```
### 1.3) Prepare the CSV file

```python
# These lines are to prepare for storing the scraped data to a csv file
csv_file = open('baklava_scrape.csv', 'w')
csv_writer = csv.writer(csv_file)
csv_writer.writerow(['baklava_name', 'baklava_size', 'baklava_price','baklava_image_url']) # The headers
```

### 1.4) Start scraping

```python
# We first find the header of the data to all the products in the website's HTML by inspecting the website 
match = soup.find("div", {"id": "ProductPageProductList"}) 

# Used to store the url to the images in the website as seen at the very top image.  
base_url = 'https://online.hafizmustafa.com'

# I found out that 'ItemOrj col-lg-3 col-md-3 col-sm-6 col-xs-6' was a class for all products, which mean
# I can loop over all the products, because of using find_all method.
for product in match.find_all('div',class_ = 'ItemOrj col-lg-3 col-md-3 col-sm-6 col-xs-6'):

    full_name = product.find('div',class_= 'productName detailUrl') # First we find the name by this class
    full_name = full_name.a.text                                    # Doing this gave us the full name of the baklava product

# All these lines of code are used to split the size from the full name, to store the size for it self in the csv file
    try:
        full_name1 = full_name.split('(')
        baklava_name = full_name1[0]
        baklava_size = full_name1[1]
    except Exception as e:
        baklava_size = None
        baklava_name = full_name

    print(baklava_name)
    print(baklava_size)


    price = product.find('div', class_ = 'discountPrice').span.text   # We found the class to the price by this line
    print(price.strip())                                            

    website_src = product.find('div', class_ = 'productImage').a.img['src'].  # Here we found the source to the image on the website, but some of
    if website_src == '/Uploads/Images/load.gif':                             # them wasn't store there as well.
        website_src = product.find('div', class_ = 'productImage').a.img['data-original'] # Therefore we added this condition to get all the sources to image
    image_url = base_url + website_src
    print(image_url)
    print()
    csv_writer.writerow([baklava_name,baklava_size,price,image_url])          # Here we did store all the scraped data into a CSV file

csv_file.close()

```
## 2.) Store all the images from URL into computer

### 2.1) Import relevant libraries and load data

<br>

<details>
<summary>Click to see the libraries</summary>

```python
# Import relevant libraries
import numpy as np
import pandas as pd
import requests
import shutil

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")

from IPython.display import display, HTML
display(HTML("<style>.container { width:78% !important; }</style>"))

# Load scraped data
df = pd.read_csv('baklava_scrape.csv')
```

</details>

<br>   

### 2.2) Store all images from URL into computer and add the path to dataframe


  ```python
  # Convert from url to jpg image on computer
  for url in df['baklava_image_url']:
      img_data = requests.get(url).content
      image_name = url.split("thumb/")[1]
      with open(image_name, 'wb') as handler:
          handler.write(img_data)
          current_dir = image_name
          move_to = f'./imagefolder/{image_name}'
          shutil.move(current_dir,move_to)

  # Add all path for the image on the repository
  base = 'imagefolder/'
  baklava_path = []
  for url_jpg in df['baklava_image_url']:
      baklava_path.append(base + url_jpg.split('thumb/')[1])

  df['baklava_image_path'] = baklava_path
  ```

### 2.3) The status of the dataframe soo far

<br>

<details>
<summary>Click to see the dataframe</summary>

<br>
<p align="center"> <img src="./git_image/dataframe_after_scraping.png" alt="Drawing"/> </p>
<br>
    
</details>

<br>  

## 3.) Data wrangling, cleaning and preproccesing 
What the dataframe looks like right now in the process can be seen right above. This section will be divided into 5 processing subsections: *Adding Premium feature*, *Cleaning the feature baklava_size and handle missing values*, *Adding Tin feature and remove whitespaces*, *Change data type of baklava_price* and *Handle the different namings on baklava_size feature*. These steps will lead us to fully preprocess the data for preparing it to visualization. The last subsection will show the final form of the dataframe.

### 3.1) Adding Premium feature

```python

# Checkpoint
df_clean = df.copy()

# Some of the baklava are Premium. Those can have their own feature called premium 
indexes_premium = df_clean[df_clean['baklava_name'].str.contains('Premium')].index
indexes_not_premium = df_clean[~df_clean['baklava_name'].str.contains('Premium')].index

#Adding the new feauture with Premium/ Not Premium
df_clean.loc[indexes_premium,'premium'] = 'Premium'
df_clean.loc[indexes_not_premium,'premium'] = 'Not Premium'

#Deleting all the places where Premium appears in name
df_clean["baklava_name"] = df_clean["baklava_name"].str.replace("Premium ", "")

```

### 3.2) Cleaning the feature baklava_size and handle missing values

```python

# We can see many ")" appears in this feature which easily can be 
# removed by following line of code 
df_clean['baklava_size'] = df_clean['baklava_size'].str.replace(')','')

# Here we would like see the observations nan appears 
df_clean[df_clean['baklava_size'].isna()]

# We actually see that the size appears in the name
df_clean.loc[df_clean['baklava_name'] == 'Ankara Walnut Baklava S Box','baklava_size'] = 'S Box'
df_clean.loc[df_clean['baklava_name'] == 'Baklava, Halep Kadayif with Pistachio L Box','baklava_size'] = 'L Box'

# Now we can remove the size from its name
df_clean['baklava_name'] = df_clean['baklava_name'].str.replace('S Box','')
df_clean['baklava_name'] = df_clean['baklava_name'].str.replace('L Box','')

```

### 3.3) Adding Tin feature and remove whitespaces

```python

# finding the indexes to material with Tin and not Tin
tin_index_in_name = df_clean[df_clean['baklava_name'] == 'HM1864 Mixed Special Metal Tin Box '].index
tin_index_in_size = df_clean[df_clean['baklava_size'].str.contains('Tin')].index
all_tin_indices = tin_index_in_name.union(tin_index_in_size)

all_index = df_clean.index
no_tin_indeces = all_index.difference(all_tin_indices)

# filling the indexes with corresponding package material
df_clean.loc[all_tin_indices,'tin'] = 'Tin'
df_clean.loc[no_tin_indeces,'tin'] = 'Not Tin'

# Remove whether the material is tin or not for all other features
df_clean['baklava_name'] = df_clean['baklava_name'].str.replace(' Metal Tin Box ','')
df_clean['baklava_size'] = df_clean['baklava_size'].str.replace(' - Tin Box','')

#Remove spaces on right and left end of all strings
df_clean['baklava_name'] = df_clean['baklava_name'].str.strip()
df_clean['baklava_size'] = df_clean['baklava_size'].str.strip()

```


### 3.4) Change data type of baklava_price

```python

# Remove newlines from the feature.
df_clean['baklava_price'] = df_clean['baklava_price'].replace('\n','', regex=True)

# Remove euro sign € from the values and add it to feature name
df_clean['baklava_price'] = df_clean['baklava_price'].replace('€','', regex=True)

#Rename the feature name
df_clean = df_clean.rename(columns={'baklava_price': 'baklava_price_euro'})
df_clean
# Change data type to float
df_clean['baklava_price_euro'] = df_clean['baklava_price_euro'].replace(',','.', regex=True)
df_clean['baklava_price_euro'] = pd.to_numeric(df_clean['baklava_price_euro'],errors='coerce')

```


### 3.5) Handle the different namings on baklava_size feature

```python
print(df_clean['baklava_size'].value_counts())

# By the code above, we can see that same things are named differently or on another language (turkish)
# Additionally we can see on all images that the packages are the same metal, so metal wil not be included.
# The tin ones are given on another feature, so we are just renaming those we have to XL, L, M etc.

# Handling the metals
df_clean['baklava_size'] = df_clean['baklava_size'].str.replace('XL Metal Box','XL')
df_clean['baklava_size'] = df_clean['baklava_size'].str.replace('L Metal Box','L')
df_clean['baklava_size'] = df_clean['baklava_size'].str.replace('M Metal Box','M')
df_clean['baklava_size'] = df_clean['baklava_size'].str.replace('S Metal Box','S')
df_clean['baklava_size'] = df_clean['baklava_size'].str.replace('XL Metal','XL')
df_clean['baklava_size'] = df_clean['baklava_size'].str.replace('S Metal','S')

# Handling the box name
df_clean['baklava_size'] = df_clean['baklava_size'].str.replace('S Box','S')
df_clean['baklava_size'] = df_clean['baklava_size'].str.replace('XL Box','XL')
df_clean['baklava_size'] = df_clean['baklava_size'].str.replace('L Box','L')

# Handling other languages and types
df_clean['baklava_size'] = df_clean['baklava_size'].str.replace('L Kutu','L')
df_clean['baklava_size'] = df_clean['baklava_size'].str.replace('S Kutu','S')
df_clean['baklava_size'] = df_clean['baklava_size'].str.replace('Large Box','L')

print(df_clean['baklava_size'].value_counts()) #approval test
```

### 3.6) Final form of the dataframe

<br>

<details>
<summary>Click to see the dataframe</summary>

<br>
<p align="center"> <img src="./git_image/webscraped_and_cleaned_dataframe.png" alt="Drawing"/> </p>
<br>
    
</details>

<br>  












