# Import packages
import pandas as pd
import plotly
import pickle
import warnings
from dash import Dash, dash_table, callback, dcc, html, Input,Output,State, ctx
import numpy as np
import mlflow
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings("ignore")
from dash.exceptions import PreventUpdate

#experiment tracking
import mlflow
import os
# This the dockerized method.
# We build two docker containers, one for python/jupyter and another for mlflow.
# The url `mlflow` is resolved into another container within the same composer.
mlflow.set_tracking_uri("http://mlflow:5000")
# In the dockerized way, the user who runs this code will be `root`.
# The MLflow will also log the run user_id as `root`.
# To change that, we need to set this environ["LOGNAME"] to your name.
os.environ["LOGNAME"] = "thamakorn"
#mlflow.create_experiment(name="chaky-diabetes-example")  #create if you haven't create
mlflow.set_experiment(experiment_name="a2_experiment_v2")



# Initialize the app
app = Dash(__name__)

#load model
loaded_model1 = pickle.load(open('/root/code/carprice_predict2.model', 'rb'))
model2 = mlflow.pyfunc.load_model('runs:/c9d107e58e6a4c13ba6749dcda021f02/model/')

scaler = pickle.load(open('/root/code/scaler.pkl', 'rb'))


app.layout = html.Div([
    html.H1("Car Price Prediction for Chaky's Company"),
    html.Div([

        html.H3(children='Instruction' , style={'textAlign':'left'}),
        html.H4("In order to predict car price, you need to choose maximum power, mileage, and  year."),
        html.H4("If you don't know maximum power and mileage, you can use provided default values to help you to predict car price."),
        html.H3(children='max power (bhp)' , style={'textAlign':'left'}),
         dcc.Input( id='max_power', type='number', value=82.4 , min=0), 

         
        html.H3(children='mileage (kmpl)', style={'textAlign':'left'}, ),
         dcc.Input( id='mileage', type='number', value=19.392 , min=0), 

         html.H3(children='year', style={'textAlign':'left'}),
         dcc.Input( id='year',type='number',value=2023, min=1983, placeholder ="select year" ),

         html.H3("Predict Car Price"),
         html.Button(n_clicks=0, id='submit', children='predict'),
         html.Br(),
         html.H4(id="y1", style={'color':'red'}),
         html.H4(id="y2", style={'color':'red'}),
        

  
        
    ])
])

@callback(
    Output("y1" ,"children"),
    Output("y2" ,"children"),
    Input('submit', 'n_clicks'),
    State('max_power', 'value'),
    State('mileage', 'value'),
    State('year', 'value'),
    prevent_initial_call = True
    
)


def update_price( n_clicks,max_power, mileage, year):
    price1 = 0
    price2 = 0
    if int(n_clicks) >= 1:
        if max_power == None:
            max_power == 82.4
        if mileage == None:
            mileage == 19.392
        if year == None:
            year == 2015
    
    
    [max_power, mileage, year]  = scaler.transform([[max_power, mileage, year]])[0]
    sample =  np.array([[max_power, mileage,year]])
    price1 = int(np.exp(loaded_model1.predict(sample)))
    intercept = np.ones([1,1])
    sample_2 = np.concatenate([intercept,sample], axis=1)
    price2 = int(np.exp(model2.predict(sample_2)))
            
    
    return f'The predicted price of old model is {price1} baht ',f'The predicted price of new model is {price2} baht'

# Run the app

if __name__ == '__main__':
    app.run(debug=True)
