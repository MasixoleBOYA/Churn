import pickle
import json
from flask import Flask, request, app,jsonify, url_for, render_template, redirect,flash,session,escape
import numpy as np
import pandas as pd

my_app = Flask(__name__)
#loading the model
model = pickle.load(open('churn_model.pkl','rb'))
@my_app.route('/')
def home():
    return render_template('home.html')

@my_app.route('/predict_api', methods =['POST'])
def predict_api():
    data= request.json['data']
    print(data)
    print(np.array(list(data.values())).reshape(1,-1))
    #new_data = model.transform(np.array(list(data.values())).reshape(1,-1))
    output = model.predict(data)
    print(output[0])
    return jsonify(output[0])
if __name__ == "__main__":
    my_app.run(debug= True)



