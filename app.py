import pickle
import re
import numpy as np
from flask import Flask, render_template, url_for, redirect, request

app = Flask(__name__)
model = pickle.load(open('irisModel.pkl', 'rb'))

@app.route("/")
def home():
    return render_template('index.html'), 'hello'

@app.route("/predict", methods=['POST'])
def predict():
    features = [int(x) for x in request.form.values()]
    features = [np.array(features)]
    prediction = model.predict(features)
    return render_template('predict.html', prediction_text=prediction)

if __name__ == '__main__':
    app.run()