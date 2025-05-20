from flask import Flask, request, render_template, jsonify
import requests
from PIL import Image
import io

app = Flask(__name__)

API_URL = 'http://localhost:5001/predict'

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'})
    
    try:
        # Gửi ảnh đến API
        files = {'file': file}
        response = requests.post(API_URL, files=files)
        
        if response.status_code == 200:
            return response.json()
        else:
            return jsonify({'error': 'API request failed'})
    
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True) 