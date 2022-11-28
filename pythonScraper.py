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
priceResultOriginal = []
descriptionResultsString = []
dublinRegionString = []
while count < 20:#10:
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

        descriptionResults = re.search('>(.*)</div>', str(description))
        addressResult = re.search('>(.*)<', str(address))

        #don't save entries that don't have an address
        if addressResult != None:
            addressResultString.append(str(addressResult.group(1)))

            mystring = str(description)
            keyword2="\"description\">"

            before_keyword2, keyword2, after_keyword2 = mystring.partition(keyword2)
            #print(after_keyword2)

            #descriptionFull = re.search('(.*)</div>', after_keyword2) #splits word from comma

            #print("---")
            keyword2="</div>"

            before_keyword2, keyword2, after_keyword2 = after_keyword2.partition(keyword2)

            descriptionResultsString.append(before_keyword2)

            # if descriptionResults != None:
            #     descriptionResultsString.append(str(descriptionResults.group(1)))
            # else:
            #     descriptionResultsString.append('')

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
                priceResultOriginal.append(str(priceResult.group(1)))
            else:
                priceResultOriginal.append('')

            numFlag = True
            priceResultNumber = re.search('€(.*) per', str(priceResult))
            priceResultType = re.search('per (.*)<!-- -->', str(priceResult))
            if (priceResultType.group(1) == 'month'):
                priceResultIntString = str(priceResultNumber.group(1))
            else:
                priceResultIntString = str(priceResultNumber.group(1))
                temp = priceResultIntString.split(',')
                priceResultIntString = str(int(''.join(temp))*4)
                numFlag = False
                priceResultString.append(priceResultIntString)

            #Remove ',' character
            if len(priceResultIntString) > 3:
                if (numFlag == True):
                    priceResultTemp1 = re.search('(.*),', priceResultIntString)
                    priceResultTemp2 = re.search(',(.*)', priceResultIntString)
                    priceResultString.append((priceResultTemp1.group(1)) + str(priceResultTemp2.group(1)))
            else:
                priceResultString.append(priceResultIntString)

            mystring = str(addressResult.group(1))
            keyword = "Dublin"

            before_keyword, keyword, after_keyword = mystring.partition(keyword) #splits address into before and after keyword "Dublin"

            if after_keyword != '': #If there is a part after the keyword, i.e., postcode exists

                wordArray = after_keyword.split() #split the address into its constituent words

                firstNum = wordArray[0] #First number found after "Dublin" keyword
                dublinRegion = re.search('(.*),', firstNum) #splits word from comma
                if dublinRegion == None: #If it returns nothing, then theres no comma
                    dublinRegionString.append(firstNum) #adds firstnum instead
                else:
                    dublinRegionString.append(dublinRegion.group(1)) #otherwise use the value without the comma
            else:
                dublinRegionString.append("Other") #otherwise its just a random other place

    url = "https://www.daft.ie/property-for-rent/dublin-city?pageSize=20&from=" + str(count*20)

columnsValues  = ['Address', 'Bedroom', 'Bathroom', 'PropertyType', 'Price']

dict = {'Address': addressResultString, 'dublinRegionString': dublinRegionString, 'Bedroom': bedResultString,'Bathroom': bathResultString,
            'PropertyType': propertyTypeResultString, 'originalPrice': priceResultOriginal, 'Price': priceResultString, 'Description': descriptionResultsString}

df = pd.DataFrame(dict)

#print(priceResultString)

# print(priceResultNumber.group(1))
# priceResultNumber = str(priceResultNumber)
#priceResultNumber.str.replace(',', '')
# print(priceResultNumber.group(1))

df.to_csv('CleanedScraperOutput.csv')
