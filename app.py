from flask import Flask, request
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
import datetime

app = Flask(__name__)

# Load the data
data = pd.read_csv('data.csv')

# Preprocess the data
X = data[['Gender', 'Height']]
y = data['Weight']

# Encode categorical features
categorical_transformer = OneHotEncoder(handle_unknown='ignore')
preprocessor = ColumnTransformer(
    transformers=[('encoder', categorical_transformer, [0])],
    remainder='passthrough'
)

X = preprocessor.fit_transform(X)

# Train the model
model = LinearRegression()
model.fit(X, y)

@app.route('/predict', methods=['POST'])
def predict():
    # Get user input from the request
    gender = request.form['gender']
    height = float(request.form['height'])

    # Preprocess the user input
    input_data = pd.DataFrame({'Gender': [gender], 'Height': [height]})
    input_data = preprocessor.transform(input_data)

    # Make predictions
    prediction = model.predict(input_data)

    # Store the prediction result in a text file
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open('prediction_results.txt', 'a') as f:
        f.write(f"{timestamp}: Gender={gender}, Height={height}, Predicted Weight={prediction[0]}\n")

    # Return the predicted weight
    return str(prediction[0])

if __name__ == '__main__':
    app.run(host='0.0.0.0')  # Make the server accessible from outside