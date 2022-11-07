import pandas as pd
from bs4 import BeautifulSoup
import requests
import re
import csv

url = "https://www.daft.ie/property-for-rent/dublin?showMap=false&sort=priceAsc"
count = 0  ##this is the number of pages you want to scrape
# the pages are in groups of 20 listings

addressResultString = []
bedResultString = []
bathResultString = []
propertyTypeResultString = []
priceResultString = []
descriptionResultsString = []
while count < 2:
    print(count)
    count = count + 1
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')

    usableLinks = []
    liData=soup.find_all('a')
    for link in liData:
        if link.has_attr('href'):
            usableLinks.append(link['href'])

    usableLinks = usableLinks[45:63]

    #44:63  usable links from the main page


    f = csv.writer(open('testing.csv ','w'))
    for link in usableLinks:
        url2 = 'https://www.daft.ie/' + link
        response = requests.get(url2)
        soup = BeautifulSoup(response.content, 'html.parser')
        description = soup.find_all('div', attrs={'data-testid':'description', 'class': 'styles__StandardParagraph-sc-15fxapi-8 eMCuSm'})
        price = soup.find_all('span', attrs={'class':'TitleBlock__StyledSpan-sc-1avkvav-5 fKAzIL'})
        address = soup.find_all('h1', attrs={'data-testid':'address'})
        bedroom = soup.find_all('p', attrs={'data-testid':'beds'})
        bathroom = soup.find_all('p', attrs={'data-testid':'baths'})
        property_type = soup.find_all('p', attrs={'data-testid':'property-type'})

        descriptionResults = re.search('>(.*)<', str(description))
        addressResult = re.search('>(.*)<', str(address))

        #don't save entries that don't have an address
        if addressResult != None:
            addressResultString.append(str(addressResult.group(1)))

            if descriptionResults != None:
                descriptionResultsString.append(str(descriptionResults.group(1)))
            else:
                descriptionResultsString.append('')

            bedResult = re.search('>(.*)Bed<', str(bedroom))
            if bedResult != None:
                bedResultString.append(str(bedResult.group(1)))
            else:
                bedResultString.append('0')

            bathResult = re.search('>(.*)Bath<', str(bathroom))
            if bathResult != None:
                bathResultString.append(str(bathResult.group(1)))
            else:
                bathResultString.append('0')

            propertyTypeResult = re.search('>(.*)<', str(property_type))
            if propertyTypeResult != None:
                propertyTypeResultString.append(str(propertyTypeResult.group(1)))
            else:
                propertyTypeResultString.append('')

            priceResult = re.search('>(.*)<!-- -->', str(price))
            if priceResult != None:
                priceResultString.append(str(priceResult.group(1)))
            else:
                priceResultString.append('')

    url = "https://www.daft.ie/property-for-rent/dublin-city?pageSize=20&from=" + str(count + 20)

columnsValues  = ['Address', 'Bedroom', 'Bathroom', 'PropertyType', 'Price']

dict = {'Address': addressResultString, 'Bedroom': bedResultString,'Bathroom': bathResultString,
            'PropertyType': propertyTypeResultString, 'Price': priceResultString, 'Description': descriptionResultsString}

df = pd.DataFrame(dict)

df.to_csv('testing.csv')
