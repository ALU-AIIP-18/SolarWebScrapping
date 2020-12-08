# Group_A_Capstone
Submission of IIIP 2020 Group A Capstone

The application used Plotly Dash and Flask as the core development environments.
It scrapes data from  the pvoutput website to create the physical asset and builds up a digital twin with ML models and solar pv output analysis and forecasting. 

The App comprises two scripts as below:


physical_asset.py- A flask App of the physical asset
The script does the following: 

1. Webscrapping to obtain physical asset data. Webscrapping limitations mean that only 12 days worth of previous data could be obtained.
2. Pre-process the data and generate a pandas dataframe.
3. Create an endpoint with the data stored as json

dig_app.py- A Dash App of the digital twin
The script does the following: 
1. Loads up the physical asset data in json type and regenerates pandas dataframe.
2. Runs the forecasting and "hinsighting" models.
3. Runs the linear regression to determining feature importance, and classification to report on performance of the model
4. Dash app layout, visualisations and interval callbacks


To run the app: 
1.physical_asset.py to perform webscrapping, preprocess the data  and create the endpoint at http://127.0.0.1:5000/
2.Run dig_app.py to access the digital twin and undertake all the modeling and forecasting. The result is an endpoint at http://127.0.0.1:8050/ which results in the application dashboard.

