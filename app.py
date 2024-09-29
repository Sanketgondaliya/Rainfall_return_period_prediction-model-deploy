from flask import Flask, request, render_template
import pickle
import numpy as np
import math
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDRegressor
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LinearRegression, SGDRegressor

app = Flask(__name__)

# Load the trained model
model_path = 'model.pkl'
model = None

try:
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
except FileNotFoundError:
    model = None
    print("Model file not found. Please ensure 'model.pkl' is available.")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return render_template('index.html', prediction_text="Model is not loaded. Please try again later.")
    
    try:
        # Extract input data from form
        form_values = list(request.form.values())
        print(form_values)
        # Convert the first input to a float (Highest_rainfall)
        highest_rainfall = float(form_values[0])
        print(highest_rainfall)
        # Compute the second input: Highest_rainfall(z) = ln(Highest_rainfall)
        highest_rainfall_z = math.log(highest_rainfall)
        print(highest_rainfall_z)
        # Prepare features for the model as a dataframe
        final_features = pd.DataFrame([[highest_rainfall, highest_rainfall_z]], 
                                      columns=['Highest_rainfall', 'Highest_rainfall(z)'])
        print(final_features)
        # Make prediction
        prediction = model.predict(final_features)
        
        # Assuming the prediction output is a single value
        output = prediction[0]

        # Return result to the user
        return render_template('index.html', prediction_text=f"Predicted Return Period for Highest Rainfall {highest_rainfall}(mm): {int(output)} Year")
    
    except ValueError:
        return render_template('index.html', prediction_text='Invalid input. Please enter valid numbers.')
    except Exception as e:
        return render_template('index.html', prediction_text=f'Error: {str(e)}')

if __name__ == "__main__":
    app.run(debug=True)
