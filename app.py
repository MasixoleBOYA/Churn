import tensorflow
import keras
import pickle
import json
import flask
from flask import Flask, request, app,jsonify, url_for, render_template, redirect,flash,session,escape
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

app = Flask(__name__)
#loading the model
model = pickle.load(open('churn_model.pkl', 'rb'))
scaler = pickle.load(open('scaling.pkl','rb'))
encoder = LabelEncoder()
minmax = MinMaxScaler()

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api', methods =['POST'])
def predict_api():
    data= request.json['data']
    print(data)

    #preprocess the data (encode and scale)
    # for key in data:
    #     if isinstance(data[key], str):
    #         data[key] = encoder.fit_transform([data[key]])[0]


    new_data = scaler.transform(np.array(list(data.values())).reshape(1,-1))

    # print(data)
    # print(np.array(list(data.values())).reshape(1,-1))
    #new_data = model.transform(np.array(list(data.values())).reshape(1,-1))
    output = model.predict(new_data)
    print(output[0])
    return jsonify(output[0])
if __name__ == "__main__":
    app.run(debug= True)


