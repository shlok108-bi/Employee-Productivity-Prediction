from flask import Flask, render_template, request
import numpy as np
import pickle
import pandas as pd
from utils import MultiColumnLabelEncoder

# Initialize Flask app
app = Flask(__name__)

# Load the saved model and label encoder
with open('gwp.pkl', 'rb') as model_file:
    model = pickle.load(model_file)
with open('label_encoder.pkl', 'rb') as encoder_file:
    mcle = pickle.load(encoder_file)

# Define categorical columns for encoding
categorical_cols = ['quarter', 'department', 'day']

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/predict')
def predict_page():
    return render_template('predict.html')

@app.route('/submit', methods=['POST'])
def predict():
    try:
        # Retrieve form data
        input_data = {
            'quarter': request.form['quarter'],
            'department': request.form['department'],
            'day': request.form['day'],
            'team': float(request.form['team']),
            'targeted_productivity': float(request.form['targeted_productivity']),
            'smv': float(request.form['smv']),
            'over_time': float(request.form['over_time']),
            'incentive': float(request.form['incentive']),
            'idle_time': float(request.form['idle_time']),
            'idle_men': float(request.form['idle_men']),
            'no_of_style_change': float(request.form['no_of_style_change']),
            'no_of_workers': float(request.form['no_of_workers']),
            'month': float(request.form['month'])
        }

        # Create DataFrame from input data
        input_df = pd.DataFrame([input_data])

        # Encode categorical variables
        input_df = mcle.transform(input_df)

        # Ensure input is in the correct order and format
        feature_order = ['quarter', 'department', 'day', 'team', 'targeted_productivity', 'smv',
                         'over_time', 'incentive', 'idle_time', 'idle_men', 'no_of_style_change',
                         'no_of_workers', 'month']
        input_array = input_df[feature_order].values

        # Make prediction
        prediction = model.predict(input_array)[0]

        # Render result on submit.html
        return render_template('submit.html', prediction=round(prediction, 4))
    except Exception as e:
        return render_template('submit.html', prediction=f"Error: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)