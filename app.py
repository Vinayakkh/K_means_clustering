from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load the K-Means model
with open('kmeans_model.pkl', 'rb') as file:
    kmeans = pickle.load(file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Extract features from the form
    annual_income = float(request.form['Annual_Income'])
    spending_score = float(request.form['Spending_Score'])
    features = np.array([[annual_income, spending_score]])
    
    # Predict the cluster
    prediction = kmeans.predict(features)[0]
    

    # Map the prediction to the respective category
    if prediction == 0:
        prediction_text = 'Customer is careless'
    elif prediction == 1:
        prediction_text = 'Customer is standard'
    elif prediction == 2:
        prediction_text = 'Customer is Target'
    elif prediction == 3:
        prediction_text = 'Customer is careful'
    else:
        prediction_text = 'Custmor is sensible'

    return render_template('index.html', prediction_text=f'Predicted Category: {prediction_text}')

if __name__ == '__main__':
    app.run(debug=True)