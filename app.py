import os
import datetime
from flask import Flask, render_template, request, jsonify
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

app = Flask(__name__)

# Load the dataset
data = pd.read_csv('GOOGL.csv')

# Limit the number of rows and columns
data = data.iloc[:10, :10]  # Limit to first 10 rows and 10 columns

# Load the trained model
if os.path.exists('trained_model.pkl'):
    loaded_model = joblib.load('trained_model.pkl')
else:
    loaded_model = None

@app.route('/')
def home():
    # Convert dataset to list of lists
    data_list = [data.columns.tolist()] + data.values.tolist()
    
    # Render a template with the dataset
    return render_template('index.html', data=data_list)

@app.route('/train_model', methods=['POST'])
def train_model():
    # Check if the model file already exists
    if os.path.exists('trained_model.pkl'):
        return 'Model already trained'
    
    # Prepare data for training
    x = data.iloc[:, 1:6]
    y = data.iloc[:, 6]
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

    # Train the model
    model = GradientBoostingRegressor()
    model.fit(X_train, y_train)

    # Generate timestamp
    timestamp = datetime.datetime.now().strftime('%Y%m%d%H%M%S')

    # Save the trained model with timestamp
    joblib.dump(model, f'trained_model_{timestamp}.pkl')

    return f'Model trained and saved as trained_model_{timestamp}.pkl'

@app.route('/predict', methods=['POST'])
def predict():
    if not loaded_model:
        return 'Model not trained yet'

    # Get the request data
    json_data = request.json
    new_data = pd.DataFrame(json_data)

    # Make predictions
    predictions = loaded_model.predict(new_data)

    # Get actual values from the test data
    y_test = data.iloc[:, 6]  # Assuming y_test is accessible in the global scope

    # Calculate metrics (optional)
    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    # Return predictions, actual values, and metrics
    return jsonify({'predicted': predictions.tolist(), 'actual': y_test.tolist(), 'metrics': {'MAE': mae, 'MSE': mse, 'R2': r2}})

if __name__ == '__main__':
    app.run(debug=True)
