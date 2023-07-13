import pickle
from flask import Flask, request, app,jsonify, url_for, render_template, redirect,flash,session,escape

import numpy as np
import pandas as pd

my_app = Flask("_name_")
model = pickle.load(open('churn_model.pkl','rb'))
@my_app.route('/')
def home():
    return render_template('home.html')

@my_app.route('/predict_api', methods =['POST'])

def predict_api():
    data= request.json['data']
    print(data)


