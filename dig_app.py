#Import libraries
import pandas as pd
import  numpy as np
import pickle
import datetime
from datetime import timedelta,date
import seaborn as sns
import matplotlib.pyplot as plt

from pvlib import solarposition, irradiance, atmosphere, pvsystem, inverter, temperature
from pvlib.forecast import GFS

import xgboost as xgb
from sklearn import metrics
from sklearn.metrics import auc, accuracy_score,  mean_squared_error,explained_variance_score,r2_score
from sklearn.model_selection import  train_test_split
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
import requests
import warnings


from pvlib.pvsystem import PVSystem, retrieve_sam
from pvlib.temperature import TEMPERATURE_MODEL_PARAMETERS
from pvlib.tracking import SingleAxisTracker
from pvlib.modelchain import ModelChain

import dash
from dash.dependencies import Input, Output, State
import dash_core_components as dcc
import dash_html_components as html
import dash_table
import plotly.express as px
import plotly.graph_objs as go
from plotly.subplots import make_subplots


response = requests.get("http://127.0.0.1:5000")
json_item=response.json()
physical_asset=pd.DataFrame.from_dict(json_item)
physical_asset['Datetime']=pd.to_datetime(physical_asset['Datetime']/1000,unit='s')
physical_asset=physical_asset.set_index('Datetime')


#For plotting previous time series
prev_dat=physical_asset.reset_index()
prev_dat=prev_dat.rename(columns={prev_dat.columns[0]:'Datetime'})

# physical_asset=pd.read_csv('physical_asset.csv',parse_dates=['Datetime']).drop_duplicates('Datetime').set_index('Datetime')
# physical_asset=physical_asset.iloc[::-1]


latitude = -28.893597
longitude = 31.468293
tz ='Africa/Johannesburg'
surface_tilt = 30
surface_azimuth = 180
albedo = 0.2

#Set beginning and end date
end=pd.Timestamp(datetime.date.today(), tz=tz) 
start = end - timedelta(12)

# Define forecast model
fm = GFS()

# Retrieve data from forecast API and perform data preparation
previous_forecast = fm.get_processed_data(latitude, longitude, start, end)
previous_forecast.index = previous_forecast.index.strftime('%Y-%m-%d %H:%M:%S')
previous_forecast.index=pd.to_datetime(previous_forecast.index)

#resample to three hours to match weather data sampling rate
data_res=physical_asset.resample('3H').mean()

#set datetime limits of solar farm data to match weather data
forecast_dates=previous_forecast.index
start_datetime=forecast_dates[0]

list_r=data_res.index
stop_datetime=list_r[-5]

date_ranges=[start_datetime,stop_datetime]
data_res=data_res[start_datetime:stop_datetime]

#Merge physical asset data with weather API data
merge_df=pd.merge(data_res,previous_forecast, how='inner', left_index=True, right_index=True)

per_hour=physical_asset['Power(W)'].groupby(physical_asset.index.time).mean().reset_index().rename(columns={physical_asset.columns[0]:'Datetime'})

# -------------------------------------------------------------------------------------

# Model

merge_dat=merge_df.reset_index()
merge_dat=merge_dat.rename(columns={merge_dat.columns[0]:'Datetime'})
X1=merge_dat.drop(columns=['Efficiency(kWh/kW)','Energy(kWh)','Datetime','Normalised(kW/kW)',
'Temperature(C)','Average(W)'])
y1=merge_dat['Energy(kWh)']

X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, test_size=0.15, random_state=42
                                                   )

xgb_model= xgb.XGBRegressor(objective="reg:linear", random_state=42)

xgb_model.fit(X1_train, y1_train)

sorted_idx = xgb_model.feature_importances_.argsort()

f_importances=pd.DataFrame({'Feature':X1.columns[sorted_idx], 'Importance':xgb_model.feature_importances_[sorted_idx]})

#-----------------------------------------------------------------
# prev_dat=merge_df.reset_index()
# prev_dat=prev_dat.rename(columns={prev_dat.columns[0]:'Datetime'})


# -------------------------------------------------------------------------------------------------------
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


#Calculate AC Power
pac=get_cast(start,end)
pac=pd.DataFrame(pac,columns = ['PVOutput'])
pac.drop(pac.tail(1).index,inplace=True)
pac.index =pac.index.strftime('%Y-%m-%d %H:%M:%S')
pac.index=pd.to_datetime(pac.index)


data=merge_df.reset_index()
data=data.rename(columns={data.columns[0]:'Datetime'})
data=data.set_index('Datetime')

#Merge two Dataframes
comparison=pd.merge(pac,data, how='inner', left_index=True, right_index=True)
comparison['Power(W)']=comparison['Power(W)']

comparison=comparison[['PVOutput','Power(W)','Efficiency(kWh/kW)']]
comparison=comparison.resample('3H').median()

#-----------------------------------------------------------------------

comparison["Performance_Factor"]=(comparison['Power(W)']-comparison['PVOutput'])/comparison['PVOutput']
comparison=comparison.replace([np.inf, -np.inf], np.nan)
comparison=comparison.fillna(0)

comparison_slice=comparison.iloc[5:,]
# create a list of our conditions
conditions = [
    (comparison_slice['Performance_Factor'] == 0)& (comparison_slice['PVOutput'] == 0),
    (comparison_slice['Performance_Factor'] <0.6) & (comparison_slice['Power(W)'] >= 0),
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
predicted=predicted[['Power(W)','Performance']]

predicted['Power(W)']=predicted['Power(W)'].round(decimals=1)
# predicted=predicted.iloc[::-1]
predicted=predicted.reset_index()
# predicted=predicted.sort_values('Power(W)')
# predicted=predicted.rename({'index':'Datetime'})
#---------------------------------------------------------------------------------------------------

start_1 = pd.Timestamp(datetime.date.today(), tz=tz) 
end_1= start_1 + timedelta(12)



pac1=get_cast(start_1,end_1)
pac1=pd.DataFrame(pac1,columns = ['PVOutput'])



#Link to external stylesheets
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css'] #https://codepen.io/chriddyp/pen/bWLwgP.css

app = dash.Dash(__name__, external_stylesheets=external_stylesheets,
meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}])

app_color = {"graph_bg": "#082255", "graph_line": "#e4ed2d"}

#_--------------------------------------------------------------------------------------------------------------

#Plots

fig = px.line(per_hour,x='index',y='Power(W)',title="Mean Hourly Power Production",
labels={"index":"Hour","Power(W)":"Production(kW)"},color_discrete_map={"Power(W)":"#fff"})

fig.update_layout(
    plot_bgcolor=app_color["graph_bg"],
    paper_bgcolor=app_color["graph_bg"],
    font={"color": "#fff"}
)
fig.update_traces(marker=dict(size=3, opacity=0.7), selector=dict(mode='marker'))



fig1 = px.pie(f_importances,values='Importance',title='Ranking of Output Predictors',
names='Feature') 
# physical_asset=physical_asset.iloc[::-1]
fig1.update_layout(
    plot_bgcolor=app_color["graph_bg"],
    paper_bgcolor=app_color["graph_bg"],
    font={"color": "#fff"}
)


fig2= px.line(prev_dat, x="Datetime", y="Power(W)", title="Previous 12 Days Production Trend", 
labels={"Power(W)":"Solar PV DC Power Output(W)"})


fig2.update_xaxes(rangeslider_visible=True,rangeselector=dict(buttons=list([dict(count=1,label="1h",step="hour",stepmode="backward"),
dict(step="all")],)))

fig2.update_traces(marker=dict(size=3, opacity=0.7), selector=dict(mode='marker'))
fig2.update_layout(
    plot_bgcolor=app_color["graph_bg"],
    paper_bgcolor=app_color["graph_bg"],
    font={"color": "#fff"}
)

#Plot comparison of actual to digital twin estimate 
x = comparison.index
y1=comparison['PVOutput']
y2=comparison['Power(W)']


fig3 = go.Figure()

fig3.add_trace(go.Scatter(
    x=x,
    y=y1,
    name = 'Predicted Pv Output'
))
fig3.add_trace(go.Scatter(
    x=x,
    y=y2,
    name='Actual Pv Output',
))

fig3.update_layout(
    title="Actual Production Vs Expected",
    plot_bgcolor=app_color["graph_bg"],
    paper_bgcolor=app_color["graph_bg"],
    font={"color": "#fff"}
)
fig3.update_xaxes(rangeslider_visible=True,rangeselector=dict(buttons=list([dict(count=1,label="1h",step="hour",stepmode="backward"),
dict(step="all")],)))



fig4 = px.line(pac1, x=pac1.index, y="PVOutput", title="12 Day Forecast",
                 range_x=[date.today(), datetime.date.today() + timedelta(12)])
fig4.update_traces(marker=dict(size=3, opacity=0.7), selector=dict(mode='marker'))
fig4.update_layout(
    plot_bgcolor=app_color["graph_bg"],
    paper_bgcolor=app_color["graph_bg"],
    font={"color": "#fff"}
)
fig4.update_xaxes(rangeslider_visible=True,rangeselector=dict(buttons=list([dict(count=1,label="1h",step="hour",stepmode="backward"),
dict(step="all")],)))



fig5 = go.Figure(go.Indicator(
    mode = "number+delta",
    value = prev_dat["Power(W)"].max(),
    delta = {"reference": 15300, "valueformat": ".0f"},
    number = {'suffix': "W","valueformat": "{:,}"},
    title = {"text": "Solar Performnace"},
    domain = {'y': [0, 1], 'x': [0.25, 0.75]}))

fig5.update_layout(
    title="Max Output Vs Rated",
    plot_bgcolor=app_color["graph_bg"],
    paper_bgcolor=app_color["graph_bg"],
    font={"color": "#fff"})

#App Layout
app.layout = html.Div(style={'backgroundColor': "#082255",'padding': 10},children=[
    html.H1(children='Solar Farm Digital Twin',className="app__header__title"),

    html.H3(children='''Application Dashboard'''),

#row 1
    html.Div([
        html.Div([

            dcc.Graph(
            id='graph1',
            figure=dict(
                layout=dict(
                plot_bgcolor=app_color["graph_bg"],
                paper_bgcolor=app_color["graph_bg"],
                                )
                            ),
                        ),

            dcc.Interval(
            id='interval-component',
            interval=600*1000, # in milliseconds
            n_intervals=0)
        ], className='six columns'),
        html.Div([

            dcc.Graph(
                id='graph2',
                figure=fig
            )
        ], className='six columns'),
    ], className='row'),

 html.Hr(),
#Row 2

    html.Div([
        html.Div([

            dcc.Graph(
                id='graph3',
                figure=fig1
            ),  
        ], className='six columns'),
        html.Div([

            dcc.Graph(
                id='graph4',
                figure=fig3
            ),
        ], className='six columns'),
    ], className='row'),
 html.Hr(),

#Row 3

    html.Div([
        html.Div([

            dcc.Graph(
                id='graph5',
                figure=fig4
            ),  
        ], className='six columns'),
        html.Div([

            dcc.Graph(figure=fig5),

            # dash_table.DataTable(
                
            #     id='table',
			# 	columns=[{"name": i, "id": i} for i in predicted.columns],
			# 	data=predicted.to_dict("rows"),
            #     style_as_list_view=True,
            #     style_header={'backgroundColor': '#082255'},
            #     style_table={'height': '600px'},
            #     style_cell={'backgroundColor': "#082255", 
            #     'color': 'white','width': 150 }
            # ),  
        ], className='six columns'),
    ], className='row')


])

@app.callback(
    Output("graph1", "figure"), [Input("interval-component", "n_intervals")]
)
def gen_figure(interval):


    fig2= px.line(prev_dat, x="Datetime", y="Power(W)", title="Previous 12 Days Production Trend",
    labels={"Power(W)":"Solar PV DC Power Output(W)"})

    fig2.update_xaxes(rangeslider_visible=True,rangeselector=dict(buttons=list([dict(count=1,
    label="1h",step="hour",stepmode="backward"),
    dict(step="all")],)))

    fig2.update_traces(marker=dict(size=3, opacity=0.7), selector=dict(mode='marker'))
    fig2.update_layout(
    plot_bgcolor=app_color["graph_bg"],
    paper_bgcolor=app_color["graph_bg"],
    font={"color": "#fff"})

    return fig2


if __name__ == '__main__':
    app.run_server(debug=True)
















