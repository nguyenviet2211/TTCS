from flask import Flask, request, jsonify
import tensorflow as tf
from PIL import Image
import numpy as np
import os
import json
from flask_cors import CORS
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.applications.vgg16 import preprocess_input
app = Flask(__name__)
CORS(app)
# Load models
models = {
    'bird': tf.keras.models.load_model('model/birdClassifierModel.h5'),
    'trash': tf.keras.models.load_model('model/trashClassifierModel.h5'),
    'emotion': tf.keras.models.load_model('model/emotionClassifierModel.h5')
}

# Load bird class indices
with open('json/bird_class_indices.json', 'r', encoding='utf-8') as f:
    bird_class_indices = json.load(f)

# Load trash class indices
with open('json/trash_class_indices.json', 'r', encoding='utf-8') as f:
    trash_class_indices = json.load(f)

with open('json/emotion_class_indicies.json', 'r', encoding='utf-8') as f:
    emotion_class_indices = json.load(f)

def preprocess_image(image, model_name):
    if model_name == 'bird':
        image = image.resize((224, 224))
        image_array = np.array(image)
        image_array = np.expand_dims(image_array, axis=0)
        image_array = preprocess_input(image_array)
    elif model_name == 'trash':
        image = image.resize((224, 224)) 
        image_array = np.array(image)
        image_array = np.expand_dims(image_array, axis=0)
        image_array = preprocess_input(image_array)
    elif model_name == 'emotion':
        image = image.resize((48, 48))
        image = image.convert('L')  
        image_array = np.array(image)
        image_array = np.expand_dims(image_array, axis=0)
        image_array = image_array / 255.0
        
    return image_array

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files.get('file')
    model_name = request.form.get('model')
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'})
    
    if model_name not in models:
        return jsonify({'error': f'Model {model_name} not found'})
    
    try:
        # Read and preprocess image
        image = Image.open(file)
        image_array = preprocess_image(image, model_name)
        
        # Make prediction using selected model
        predictions = models[model_name].predict(image_array)
        
        # Get top prediction
        class_idx = np.argmax(predictions[0])
        confidence = float(predictions[0][class_idx])
        print(predictions)
        # Lấy tên class nếu là bird
        if model_name == 'bird':
            class_name = bird_class_indices[str(class_idx)]
        elif model_name == 'trash':
            class_name = trash_class_indices[str(class_idx)]
        elif model_name == 'emotion':
            class_name = emotion_class_indices[str(class_idx)]
        
        return jsonify({
            'model': model_name,
            'class': class_name,
            'confidence': confidence
        })
    
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(port=5001, debug=True) 