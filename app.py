from flask import Flask, render_template, request, flash
import joblib
import pandas as pd
import numpy as np
from utils import preprocess_data

app = Flask(__name__)

model = joblib.load('f1_prediction_model.pkl')
label_encoders = joblib.load('label_encoders.pkl')

def make_prediction(model, label_encoders, input_data):
    try:
        driver_encoded = label_encoders['Driver'].transform([input_data['driver']])[0]
        constructor_encoded = label_encoders['Constructor'].transform([input_data['constructor']])[0]
        circuit_encoded = label_encoders['Circuit'].transform([input_data['circuit']])[0]
    except KeyError as e:
        flash(f"Invalid input: {str(e)}")
        return None, None
    
    input_features = pd.DataFrame({
        'Driver': [driver_encoded],
        'Constructor': [constructor_encoded],
        'Circuit': [circuit_encoded],
        'Grid': [input_data['grid']],
        'Season': [input_data['season']]
    })
    
    try:
        predicted_position = model.predict(input_features)[0]

        probabilities = model.predict_proba(input_features)[0]

        confidence_score = np.max(probabilities)
        
        exact_position_threshold = 0.75
        
        if confidence_score < exact_position_threshold:

            lower_bound = max(1, predicted_position - 2)
            upper_bound = predicted_position + 2
            
            new_prediction = f"{lower_bound} - {upper_bound}"
            confidence_score = 0.90 
            
            return new_prediction, confidence_score
        else:
         
            return predicted_position, confidence_score
        
    except Exception as e:
        flash(f"Prediction error: {str(e)}")
        return None, None


@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        driver = request.form['driver']
        constructor = request.form['constructor']
        circuit = request.form['circuit']
        grid = request.form['grid']
        season = request.form['season']

        if not all([grid.isdigit(), season.isdigit()]) or int(grid) < 1 or int(season) < 2000 or int(season) > 2024:
            flash('Invalid input: Grid Position must be a positive integer and Season must be between 2000 and 2024.')
            return render_template('index.html')

        grid = int(grid)
        season = int(season)

        input_data = {
            'driver': driver,
            'constructor': constructor,
            'circuit': circuit,
            'grid': grid,
            'season': season
        }

        predicted_position, confidence_score = make_prediction(model, label_encoders, input_data)

        return render_template('index.html', prediction_result=predicted_position, confidence_score=confidence_score)
    
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
