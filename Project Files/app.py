import numpy as np
import pickle
import pandas as pd
import os
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

# Load trained model and scaler
model = pickle.load(open('model.pkl', 'rb'))
scale = pickle.load(open('encoder.pkl', 'rb'))  # Ensure this was trained with correct feature names

@app.route('/')  # Home page
def home():
    return render_template("index.html")  # Rendering the home page


@app.route('/predict', methods=["POST", "GET"])  # Prediction route
def predict():
    try:
        # Read input values and ensure all expected features are present
        input_feature = [float(x) for x in request.form.values()]
        
        # Define correct feature order
        categorical_columns = ['holiday', 'temp', 'rain', 'snow', 'weather', 'year', 'month', 'day','hours', 'minutes', 'seconds']

        # Ensure the number of input features matches expected features
        if len(input_feature) != len(categorical_columns):
            return jsonify({"error": "Feature mismatch. Expected {}, but got {}.".format(len(), len(input_feature))})
        
        # Convert to DataFrame
        data = pd.DataFrame([input_feature], columns=categorical_columns)
        print("Feature names in input data:", data.columns)  # Debugging step

        # Transform input data
        data = scale.transform(data)

        # Predict using the trained model
        prediction = model.predict(data)

        # Return the result
        return render_template("ouput.html", prediction_text="Estimated Traffic Volume: " + str(prediction[0]))

    except Exception as e:
        return jsonify({"error": str(e)})


if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(port=port, debug=True, use_reloader=False)