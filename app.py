from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)

# Load the trained model from model.pkl file
rf_classifier = joblib.load('model.pkl')

# Load the label encoder from label_encoder.pkl file
label_encoder = joblib.load('label_encoder.pkl')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    features = [float(x) for x in request.form.values()]
    features = np.array(features).reshape(1, -1)

    # Make the prediction
    prediction_index = rf_classifier.predict(features)[0]
    predicted_species = label_encoder.inverse_transform([prediction_index])[0]

    return render_template('index.html', prediction_text='Predicted Fish Species: {}'.format(predicted_species))


if __name__ == '__main__':
    app.run(debug=True)
