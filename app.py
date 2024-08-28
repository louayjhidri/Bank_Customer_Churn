from flask import Flask, render_template, request
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from Customer_churn.components.model_prediction import CustomData, ModelPrediction
from Customer_churn.config.configuration import ConfigurationManager
import dill
import os
application = Flask(__name__)
app=application
#### Route for a home page

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