from flask import Flask, request, render_template
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

application = Flask(__name__)
app = application

## Route for a home page
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        # Collect form data
        data = CustomData(
            blood_urea=float(request.form.get('blood_urea')),
            white_blood_cell_count=float(request.form.get('white_blood_cell_count')),
            blood_glucose_random=float(request.form.get('blood_glucose_random')),
            serum_creatinine=float(request.form.get('serum_creatinine')),
            albumin=float(request.form.get('albumin')),
            hypertension=request.form.get('hypertension')
        )

        # Convert data to DataFrame for prediction
        pred_df = data.get_data_as_data_frame()

        # Create prediction pipeline instance
        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(pred_df)

        # Round the result to 2 decimal places
        rounded_result = round(results[0], 2)

        # Convert the result to string based on the condition
        if rounded_result >= 0.5:
            result_string = "Not Chronic Kidney Disease"
        else:
            result_string = "Chronic Kidney Disease"

        # Pass the result_string to the template
        return render_template('home.html', results=result_string)

if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)
