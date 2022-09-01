import pickle
from flask import Flask, request, app, jsonify, url_for, render_template
import numpy as np
import pandas as pd

app = Flask(__name__) # Staring a Flask app

model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('home.html')

#  Creating an API
@app.route("/predict", methods=['POST'])
def predict():
    # data = request.json['data']
    data = [float(x) for x in request.form.values()]
    print(data)
    # new_data = [list(data.values())]
    new_data = [data]
    output = model.predict(new_data)[0]
    print(output)
    # return jsonify(output)
    return render_template('home.html', prediction_text="Airfoil Pressure {}".format(output))

if __name__ == "__main__":
    app.run(debug=True)
