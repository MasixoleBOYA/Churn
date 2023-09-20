import tensorflow
from tensorflow import keras
from keras.models import load_model
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
#model = pickle.load(open('churn_model.pkl', 'rb'))
model = load_model('churn_model.h5')

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

@app.route('/predict',methods = ['POST'])
def predict():
    data = [float(x) for x in request.form.values()]
    final_input = scaler.transform(np.array(data).reshape(1,-1))
    print(final_input)
    output = model.predict(final_input)[0]
    return render_template("home.html", prediction_text = "The churn estimation is {}".format(output))

if __name__ == "__main__":
    app.run(debug= True)


