#importing all the necessary libraries
import pickle
import re
import numpy as np
from flask import Flask, render_template, url_for, redirect, request

app = Flask(__name__)
#loading the model
model = pickle.load(open('irisModel.pkl', 'rb'))

#app home page
@app.route("/")
def home():
    return render_template('index.html'), 'hello'

#app prediction page
@app.route("/predict", methods=['POST'])
def predict():
    #taking values of the features from form
    features = [int(x) for x in request.form.values()]
    features = [np.array(features)]
    prediction = model.predict(features)
    return render_template('predict.html', prediction_text=prediction)

if __name__ == '__main__':
    app.run()