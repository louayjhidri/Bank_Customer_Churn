from flask import Flask, render_template, request
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from Customer_churn.components.model_prediction import CustomData, ModelPrediction
from Customer_churn.config.configuration import ConfigurationManager
import dill
import os

from dash import Dash,Input, Output, dcc, html
import dash_bootstrap_components as dbc
import plotly.express as px
application = Flask(__name__)
app=application

dash_app = Dash(__name__, server=application, url_base_pathname='/dashboard/', external_stylesheets=[dbc.themes.BOOTSTRAP])

# Load data and create visualizations for Dash (example)
data = pd.read_csv('artifacts/data_ingestion/churn/Customer-Churn-Records.csv')

# Define custom color sequences
color_sequence = ['#1f77b4', '#ff7f0e']  # Blue and Orange
color_sequence_high_risk = ['#d62728', '#2ca02c']  # Red and Green

# Create visualizations
fig_churn_rate = px.pie(
    names=['Churned', 'NotChurned'],
    values=[data['Exited'].sum(), len(data) - data['Exited'].sum()],
    title='Churn Rate',
    color_discrete_sequence=['#636EFA', '#EF553B']  # Blue and Red
)

fig_age_churn = px.histogram(
    data,
    x='Age',
    color='Exited',
    title='Churn by Age',
    color_discrete_sequence=color_sequence
)

fig_gender_churn = px.histogram(
    data,
    x='Gender',
    color='Exited',
    title='Churn by Gender',
    color_discrete_sequence=color_sequence
)

fig_credit_churn = px.histogram(
    data,
    x='CreditScore',
    color='Exited',
    title='Churn by Credit Score',
    color_discrete_sequence=color_sequence
)

fig_loyalty = px.histogram(
    data,
    x='SatisfactionScore',
    color='Exited',
    title='Customer Loyalty Segmentation',
    color_discrete_sequence=color_sequence
)

fig_high_risk = px.histogram(
    data[(data['SatisfactionScore'] < 3) & (data['IsActiveMember'] == 0)],
    x='Tenure',
    color='Exited',
    title='High-Risk Segments',
    color_discrete_sequence=color_sequence_high_risk
)

fig_satisfaction = px.histogram(
    data,
    x='SatisfactionScore',
    title='Satisfaction Score Distribution',
    color_discrete_sequence=['#1f77b4']  # Blue
)

fig_corr_satisfaction = px.scatter(
    data,
    x='SatisfactionScore',
    y='Exited',
    trendline="ols",
    title='Satisfaction vs Churn',
    color_discrete_sequence=['#1f77b4']  # Blue
)

fig_active_inactive = px.histogram(
    data,
    x='IsActiveMember',
    color='Exited',
    title='Active vs Inactive Members',
    color_discrete_sequence=color_sequence
)

fig_financial_health = px.scatter(
    data,
    x='CreditScore',
    y='Balance',
    color='Exited',
    title='Financial Health Indicators',
    color_discrete_sequence=color_sequence
)

# Define the layout of the Dash app
dash_app.layout = html.Div([
    
    
    
    dbc.Row([
        dbc.Col(dcc.Graph(figure=fig_churn_rate), width=6),
        dbc.Col(dcc.Graph(figure=fig_age_churn), width=6)
    ]),
    dbc.Row([
        dbc.Col(dcc.Graph(figure=fig_gender_churn), width=6),
        dbc.Col(dcc.Graph(figure=fig_credit_churn), width=6)
    ]),
    dbc.Row([
        dbc.Col(dcc.Graph(figure=fig_loyalty), width=6),
        dbc.Col(dcc.Graph(figure=fig_high_risk), width=6)
    ]),
    dbc.Row([
        dbc.Col(dcc.Graph(figure=fig_satisfaction), width=6),
        dbc.Col(dcc.Graph(figure=fig_corr_satisfaction), width=6)
    ]),
    dbc.Row([
        dbc.Col(dcc.Graph(figure=fig_active_inactive), width=6),
        dbc.Col(dcc.Graph(figure=fig_financial_health), width=6)
    ])
])


#### Route for a home page
@application.route('/dashboard_page')
def dashboard_page():
    return render_template('dashboard.html')
@app.route('/')
def index():
    return render_template('index.html')
@app.route('/predict',methods=['GET','POST'])
def predict_datapoint():
    if request.method=='GET':
        return render_template('home.html')
    else:
        data = CustomData(
            CreditScore=int(request.form.get('CreditScore')),
            Gender=request.form.get('Gender'),
            Age=int(request.form.get('Age')),
            Tenure=int(request.form.get('Tenure')),
            Balance=float(request.form.get('Balance')),
            NumOfProducts=int(request.form.get('NumOfProducts')),
            HasCrCard=int(request.form.get('HasCrCard')),
            IsActiveMember=int(request.form.get('IsActiveMember')),
            EstimatedSalary=float(request.form.get('EstimatedSalary')),
            SatisfactionScore=int(request.form.get('SatisfactionScore')),
            CardType=request.form.get('CardType'),
            PointEarned=int(request.form.get('PointEarned')),




        )
        pred_df=data.get_data_as_dataframe()
        print(pred_df)
        config = ConfigurationManager()
        model_predicition_config = config.get_model_prediction_config()
        modelprediction=ModelPrediction(model_prediction_config=model_predicition_config)
        results=modelprediction.predict(pred_df)
        print( "results : ",results)
        if results[0]==0:
            return render_template('home.html',results="Not Churned")
        else:
            return render_template('home.html',results="Churned")
        

if __name__=="__main__":
    app.run(host="0.0.0.0",debug=True)