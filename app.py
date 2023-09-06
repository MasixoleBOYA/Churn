import tensorflow
import keras
import pickle
import json
import flask
from flask import Flask, request, app,jsonify, url_for, render_template, redirect,flash,session,escape
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

import sys
sys.path.append('C:/Users/STAFF/Desktop/Git_Repos/Customer_churn')

app = Flask(__name__)
#loading the model
model = pickle.load(open('churn_model.pkl', 'rb'))
scaler = pickle.load(open('scaling.pkl','rb'))
from churn import X_train

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api', methods =['POST'])
def predict_api():
    data= request.json['data']
    print(data)
    scaler.fit(X_train)
    # data = list(data.values())
    new_data = scaler.transform(np.array(list(data.values())).reshape(1,-1))
    output = model.predict(new_data)
    output = output.tolist()

    print(output[0])
    return jsonify(output[0])
if __name__ == "__main__":
    app.run(debug= True)


