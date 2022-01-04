#Import libraries
import pandas as pd
import  numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import xgboost as xgb
from sklearn import metrics
from sklearn.metrics import auc, accuracy_score,  mean_squared_error,explained_variance_score,r2_score
from sklearn.model_selection import  train_test_split
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
import requests
import warnings

from pvlib import solarposition, irradiance, atmosphere, pvsystem, inverter, temperature
from pvlib.forecast import GFS
from pvlib.pvsystem import PVSystem, retrieve_sam
from pvlib.temperature import TEMPERATURE_MODEL_PARAMETERS
from pvlib.tracking import SingleAxisTracker
from pvlib.modelchain import ModelChain
import plotly.express as px
import plotly.graph_objs as go
import altair as alt
from plotly.subplots import make_subplots

# import bs4
from bs4 import BeautifulSoup #l tool for parsing html data
import datetime as datetime
from datetime import date, timedelta
import streamlit as st

######################################################################################################



######################################################################################################

st.title("Solar Farm Digital Twin")
st.markdown("""
This application illustrates the application of digital twin technology to a grid-connected solar PV 
installation. Web-Scrapped data from a live solar pv installation simulates sensors attached to the physical asset.
The data is then compared against a simulation of the installation farm in order to evaluate performance of the real plant, and also undertake 
forecasting at 3 hour intervals. The simulation is achieved using the PVlib Library.
""")
@st.cache
######################################################################################################
def get_data():
  dt = date.today() - timedelta(12)
  lst={d.strftime('%Y%m%d') for d in pd.date_range(dt,date.today())}

  #the base url address with a blank 'dt' tag that will be egenrate from a loop
  url_test = 'https://pvoutput.org/intraday.jsp?id=49819&sid=45346&dt=%s'

  # List comprehension to geenrate urls
  urls=[url_test %i for i in lst]
  urls.sort(key=lambda url: url.split('/')[-1], reverse=False)

  #####################################################################################################
  #generate pages usign requests
  pages=[requests.get(j)for j in urls]

  #use beautiful soup to parse html
  soups=[BeautifulSoup(page.text, 'html.parser') for page in pages]

  #####################################################################################################
  # This workflow extracts data from the datatables and generates a list of dataframes, one for each day
  tables=[soup.find('table',id='tb') for soup in soups]

  table_rows = [table.find_all('tr') for table in tables]

  results=[]
  for table_row in table_rows:
    res=[]
    for tr in table_row:
      td = tr.find_all('td')
      row = [tr.text.strip() for tr in td if tr.text.strip()]
      if row:
          res.append(row)
    results.append(res)

  # Generate list of dataframes
  dfs = [pd.DataFrame(i, columns=['Date','Time','Energy(kWh)','Efficiency(kWh/kW)','Power(W)','Average(W)',
        'Normalised(kW/kW)','Temperature(C)','Voltage(V)','Energy Used(kWh)','Power Used(W)']) for i in results ]
  #Remove first row which picked erroneous data
  dfs=[df[1:] for df in dfs]  
  # Concatenate list of dataframes into single df
  data=pd.concat(dfs)
  # Preprocessign data
# Removes "W", 'kWh' labels and thousand separator commas
  data['Energy(kWh)'] = data['Energy(kWh)'].str.replace('kWh', '')
  data['Efficiency(kWh/kW)'] = data['Efficiency(kWh/kW)'].str.replace('kWh/kW', '')
  data['SolarOutput(kW)'] = data['Power(W)'].str.replace('W', '').str.replace(',', '')
  data['SolarOutput(kW)'] =data['SolarOutput(kW)'].apply(pd.to_numeric, errors='coerce').multiply(0.001)


  data['SolarPowerAverage(kW)'] =data['Average(W)'].str.replace('W', '')
  data['SolarPowerAverage(kW)']=data['SolarPowerAverage(kW)'].apply(pd.to_numeric, errors='coerce').multiply(0.001)

  data['Normalised(kW/kW)'] = data['Normalised(kW/kW)'].str.replace('kW/kW', '')
  data['Temperature(C)'] = data['Temperature(C)'].str.replace('C', '')

  # data['Voltage(V)'] = data['Voltage(V)'].str.replace('-', 0)
  data['Energy Used(kWh)'] = data['Energy Used(kWh)'].str.replace('kWh', '')
  data['PowerUsed(kW)'] = data['Power Used(W)'].str.replace('W', '').str.replace(',', '')
  data['PowerUsed(kW)'] =data['PowerUsed(kW)'].apply(pd.to_numeric, errors='coerce').multiply(0.001)

  data["Date"]=pd.to_datetime(data['Date'], format='%d/%m/%y') 
  data["Date"]=data["Date"].astype(str)
  # #Combine Date and Time Columns and convert to Datetime
  data['Time']= pd.to_datetime(data['Time']).dt.strftime('%H:%M:%S')
  data['Datetime'] =pd.to_datetime(data['Date'] + ' ' + data['Time'])

  # # add date as string column

  data.drop(['Date','Time'],axis=1,inplace=True)

  cols=data.columns.drop(['Datetime'])
  data[cols] = data[cols].apply(pd.to_numeric, errors='coerce')


  # #Reorder Columns
  data=data[['Datetime','Energy(kWh)', 'Efficiency(kWh/kW)', 'SolarOutput(kW)', 'SolarPowerAverage(kW)',
        'Normalised(kW/kW)', 'Temperature(C)', 'Voltage(V)', 'Energy Used(kWh)',
        'PowerUsed(kW)']]


  data.drop('Voltage(V)',axis=1,inplace=True)



  data=data.fillna(0)
  data=data.sort_values(by=['Datetime'],ascending=True)
  data['Import/Export']=data['Energy(kWh)']-data['Energy Used(kWh)']
  data=data.set_index('Datetime')
  return data
data=get_data()
############################################################################

############################################################################




#######################################################################################


#############################################################################

latitude = -28.89367
longitude = 31.46824

# latitude = -28.893597
# longitude = 31.468293

tz ='Africa/Johannesburg'
surface_tilt = 30
surface_azimuth = 180
albedo = 0.2

#Set beginning and end date
end=pd.Timestamp(datetime.date.today(), tz=tz) 
start = end-timedelta(12)

# Define forecast model
fm = GFS()

# Retrieve data from forecast API and perform data preparation
previous_forecast = fm.get_data(latitude, longitude, start, end)
previous_forecast.index = previous_forecast.index.strftime('%Y-%m-%d %H:%M:%S')
previous_forecast.index=pd.to_datetime(previous_forecast.index)

#resample to three hours to match weather data sampling rate
data_res=data.resample('3H',offset = '2H').mean()

#set datetime limits of solar farm data to match weather data
forecast_dates=previous_forecast.index
start_datetime=forecast_dates[0]

list_r=data_res.index
stop_datetime=list_r[-5]

date_ranges=[start_datetime,stop_datetime]
data_res=data_res[start_datetime:stop_datetime]

#Merge physical asset data with weather API data
merge_df=pd.merge(data_res,previous_forecast, how='inner', left_index=True, right_index=True)

per_hour=data['SolarOutput(kW)'].groupby(data.index.time).mean().reset_index().rename(columns={data.columns[0]:'Datetime'})

##################################################################################

# Model

merge_dat=merge_df.reset_index()
merge_dat=merge_dat.rename(columns={merge_dat.columns[0]:'Datetime'})
X1=merge_dat.drop(columns=['Efficiency(kWh/kW)','Energy(kWh)','Datetime','Normalised(kW/kW)',
'Temperature(C)','SolarPowerAverage(kW)'])
y1=merge_dat['Energy(kWh)']

X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, test_size=0.15, random_state=42
                                                   )

xgb_model= xgb.XGBRegressor(objective="reg:linear", random_state=42)

xgb_model.fit(X1_train, y1_train)

sorted_idx = xgb_model.feature_importances_.argsort()

f_importances=pd.DataFrame({'Feature':X1.columns[sorted_idx], 'Importance':xgb_model.feature_importances_[sorted_idx]})

f_importances=f_importances.sort_values(by="Importance",ascending=False)


##############################################################################

def get_cast(start_date,end_date):    
    from pvlib.pvsystem import PVSystem, retrieve_sam

    from pvlib.temperature import TEMPERATURE_MODEL_PARAMETERS

    from pvlib.tracking import SingleAxisTracker

    from pvlib.modelchain import ModelChain

    sandia_modules = retrieve_sam('sandiamod')

    cec_inverters = retrieve_sam('cecinverter')

    module = sandia_modules['SolarWorld_Sunmodule_250_Poly__2013_']

    inverter = cec_inverters['ABB__TRIO_20_0_TL_OUTD_S1_US_480__480V_']

    temperature_model_parameters = TEMPERATURE_MODEL_PARAMETERS['sapm']['open_rack_glass_glass']

    # model a single axis tracker
    system = SingleAxisTracker(module_parameters=module, inverter_parameters=inverter, temperature_model_parameters=temperature_model_parameters, modules_per_string=15, strings_per_inverter=4)

    # fx is a common abbreviation for forecast
    fx_model = GFS()

    forecast_mod = fx_model.get_processed_data(latitude, longitude, start_date, end_date)

    # use a ModelChain object to calculate modeling intermediates
    mchain = ModelChain(system, fx_model.location)

    # extract relevant data for model chain
    mchain.run_model(forecast_mod)
    acp=mchain.ac.fillna(0)
    return acp
##################################################################################
#Calculate AC Power
pac=get_cast(start,end)*0.001
pac=pd.DataFrame(pac,columns = ['PVOutput'])

pac.drop(pac.tail(1).index,inplace=True)
pac.index =pac.index.strftime('%Y-%m-%d %H:%M:%S')
pac.index=pd.to_datetime(pac.index)


data=merge_df.reset_index()
data=data.rename(columns={data.columns[0]:'Datetime'})
data=data.set_index('Datetime')

#Merge two Dataframes
comparison=pd.merge(pac,data, how='inner', left_index=True, right_index=True)
comparison['SolarPowerAverage(kW)']=comparison['SolarPowerAverage(kW)']

comparison=comparison[['PVOutput','SolarPowerAverage(kW)','Efficiency(kWh/kW)']]
comparison=comparison.resample('3H').median()

#-----------------------------------------------------------------------

comparison["Performance_Factor"]=(comparison['SolarPowerAverage(kW)']-comparison['PVOutput'])/comparison['PVOutput']
comparison=comparison.replace([np.inf, -np.inf], np.nan)
comparison=comparison.fillna(0)

comparison_slice=comparison.iloc[5:,]
# create a list of our conditions
conditions = [
    (comparison_slice['Performance_Factor'] == 0)& (comparison_slice['PVOutput'] == 0),
    (comparison_slice['Performance_Factor'] <0.6) & (comparison_slice['SolarPowerAverage(kW)'] >= 0),
    (comparison_slice['Performance_Factor'] >0.6) & (comparison_slice['Performance_Factor'] < 1),
    (comparison_slice['Performance_Factor'] > 1)
    ]

# create a list of the values we want to assign for each condition
values = ['night', 'underperforming', 'normal', 'oveperforming']

# create a new column and use np.select to assign values to it using our lists as arguments
comparison_slice['Class'] = np.select(conditions, values)

le = preprocessing.LabelEncoder()


predictors=comparison_slice.drop(columns=['Class'])
target=comparison_slice['Class']


target=le.fit_transform(target)

X2_train, X2_test, y2_train, y2_test = train_test_split(predictors,target, test_size=0.15, random_state=42
                                                   )

rf =RandomForestClassifier(max_depth=4, random_state=0)

rf.fit(X2_train, y2_train)

#predict on last 5 readings


comparison=comparison.iloc[::-1]
predicted=comparison.iloc[:5,:].reset_index()
new_predictions=y2_pred=rf.predict(predicted.iloc[:5,1:])
new_predictions=le.inverse_transform(new_predictions)

predicted['Performance']=new_predictions
predicted=predicted.set_index('index')
predicted.index.rename('Datetime',inplace=True)
predicted=predicted[['SolarPowerAverage(kW)','Performance']]

predicted['SolarPowerAverage(kW)']=predicted['SolarPowerAverage(kW)'].round(decimals=1)
# predicted=predicted.iloc[::-1]
predicted=predicted.reset_index()
# predicted=predicted.sort_values('Power(W)')
# predicted=predicted.rename({'index':'Datetime'})
#---------------------------------------------------------------------------------------------------

start_1 = pd.Timestamp(datetime.date.today(), tz=tz) 
end_1= start_1 + timedelta(12)



pac1=get_cast(start_1,end_1)*0.001
pac1=pd.DataFrame(pac1,columns = ['PVOutput'])
# st.dataframe(pac1)


##############################################################################


colz1,colz2,colz3,colz4=st.columns(4)

#Plot select metrics
with colz1:
    MaxPower=st.metric(label="Max Solar Power(kW)",value=round(data['SolarPowerAverage(kW)'].max(),2))
with colz2:
    CapacityFactor=st.metric(label="Capacity Factor",value=round((data['SolarOutput(kW)'].median()/15.3),2))
with colz3:
    MedianExport=st.metric(label="Median Export(Import)(kW)",value=round(data['Import/Export'].max(),2))
with colz4:
    SystemEfficiency=st.metric(label="Module Efficiency (kwh/kW)",value=round(data['Efficiency(kWh/kW)'].max(),2))


####################################################################
fig3 = go.Figure()

fig3.add_trace(go.Scatter(
    x=data.index,
    y=data['SolarPowerAverage(kW)']
))

fig3.update_layout(
    title="12-Day Plant Performance",
    width=800,
    height=600
)
fig3.update_xaxes(rangeslider_visible=True,rangeselector=dict(buttons=list([dict(count=1,label="1h",step="hour",stepmode="backward"),
dict(step="all")],)))

st.write(fig3)

##################################################################
cols1,cols2=st.columns(2)

#Plot select metrics
with cols1:
    c=alt.Chart(f_importances.sort_values(by='Importance',ascending=False)).mark_bar(opacity=0.7, color='#FF0000').encode(
    x='Feature',
    y='Importance'
).properties(
    width=350,
    height=450,title='Ranking of Feature Importances')
    FeatureImportance=st.write(c)
with cols2:
    CapacityFactor=st.metric(label="Capacity Factor",value=round((data['SolarOutput(kW)'].median()/15.3),2))
