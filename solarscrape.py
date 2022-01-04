import bs4
from bs4 import BeautifulSoup #l tool for parsing html data
import requests # for making standard html requests
import pickle
import pandas as pd
#import date utils
import datetime as datetime
from datetime import date, timedelta


# The daily data is stored in data tables on the website which are differentiated by the 'dt' tag at the end, which contains the date
# See here https://pvoutput.org/intraday.jsp?id=49819&sid=45346&dt=20201110
# The process below generates dates starting from today going backwards 12 days, and add the date to the 'dt' tag
#subtract 12 days

dt = date.today() - timedelta(12)
lst = {d.strftime('%Y%m%d') for d in pd.date_range(dt, date.today())}

#the base url address with a blank 'dt' tag that will be egenrate from a loop
url_test = 'https://pvoutput.org/intraday.jsp?id=49819&sid=45346&dt=%s'


# List comprehension to generate urls
urls = [url_test % i for i in lst]
urls.sort(key=lambda url: url.split('/')[-1], reverse=False)

#generate pages usign requests
pages = [requests.get(j)for j in urls]

#use beautiful soup to parse html
soups = [BeautifulSoup(page.text, 'html.parser') for page in pages]


# This workflow extracts data from the datatables and generates a list of dataframes, one for each day
tables = [soup.find('table', id='tb') for soup in soups]

table_rows = [table.find_all('tr') for table in tables]


rests = []
for table_row in table_rows:
  res = []
  for tr in table_row:
    td = tr.find_all('td')
    row = [tr.text.strip() for tr in td if tr.text.strip()]
    if row:
        res.append(row)
  rests.append(res)

# Generate list of dataframes
dfs = [pd.DataFrame(i, columns=['Date', 'Time', 'Energy(kWh)', 'Efficiency(kWh/kW)', 'Power(W)', 'Average(W)',
                                'Normalised(kW/kW)', 'Temperature(C)', 'Voltage(V)', 'Energy Used(kWh)', 'Power Used(W)']) for i in rests]
#Remove first row which picked erroneous data
dfs = [df[1:] for df in dfs]

# Concatenate list of dataframes into single df
data = pd.concat(dfs)



# Preprocessign data
# Removes "W", 'kWh' labels and thousand separator commas
data['Energy(kWh)'] = data['Energy(kWh)'].str.replace('kWh', '')
data['Efficiency(kWh/kW)'] = data['Efficiency(kWh/kW)'].str.replace('kWh/kW', '')
data['Power(W)'] = data['Power(W)'].str.replace('W', '').str.replace(',', '')

data['Average(W)'] = data['Average(W)'].str.replace('W', '')
data['Normalised(kW/kW)'] = data['Normalised(kW/kW)'].str.replace('kW/kW', '')
data['Temperature(C)'] = data['Temperature(C)'].str.replace('C', '')

# data['Voltage(V)'] = data['Voltage(V)'].str.replace('-', 0)
data['Energy Used(kWh)'] = data['Energy Used(kWh)'].str.replace('kWh', '')
data['Power Used(W)'] = data['Power Used(W)'].str.replace(
    'W', '').str.replace(',', '')
data["Date"] = pd.to_datetime(data['Date'], format='%d/%m/%y')
data["Date"] = data["Date"].astype(str)
# #Combine Date and Time Columns and convert to Datetime
data['Time'] = pd.to_datetime(data['Time']).dt.strftime('%H:%M:%S')
data['Datetime'] = pd.to_datetime(data['Date'] + ' ' + data['Time'])


# # add date as string column

data.drop(['Date', 'Time'], axis=1, inplace=True)

cols = data.columns.drop(['Datetime'])
data[cols] = data[cols].apply(pd.to_numeric, errors='coerce')


# #Reorder Columns
data = data[['Datetime', 'Energy(kWh)', 'Efficiency(kWh/kW)', 'Power(W)', 'Average(W)',
             'Normalised(kW/kW)', 'Temperature(C)', 'Voltage(V)', 'Energy Used(kWh)',
            'Power Used(W)']]


data.drop('Voltage(V)', axis=1, inplace=True)


data = data.fillna(0)
data = data.sort_values(by=['Datetime'], ascending=False)
print(data.head())
# # data.to_csv('physical_asset.csv',index=False,mode='a',header=False)


# @app.route("/")
# def dfjson():
#     """
#     return a json representation of the dataframe
#     """

#     return Response(data.to_json(orient="records"), mimetype='application/json')


# if __name__ == "__main__":

#     app.run(debug=True)
