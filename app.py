from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

# Load the model and label encoder
rf_classifier = joblib.load('model.pkl')
label_encoder = joblib.load('label_encoder.pkl')


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get the input data from the form
        Weight = float(request.form['Weight'])
        Length1 = float(request.form['Length1'])
        Length2 = float(request.form['Length2'])
        Length3 = float(request.form['Length3'])
        Height = float(request.form['Height'])
        Width = float(request.form['Width'])

        # Make a prediction using the model
        prediction = rf_classifier.predict(
            [[Weight, Length1, Length2, Length3, Height, Width]])
        predicted_species = label_encoder.inverse_transform(prediction)[0]

        # Prepare the input data dictionary to pass to the template
        input_data = {
            'Weight': Weight,
            'Length1': Length1,
            'Length2': Length2,
            'Length3': Length3,
            'Height': Height,
            'Width': Width
        }

        # Render the template with the prediction result and input data
        return render_template('index.html', prediction_text=predicted_species, input_data=input_data)

    # Render the initial form without any prediction
    return render_template('index.html', prediction_text=None)


if __name__ == "__main__":
    app.run(debug=True)
