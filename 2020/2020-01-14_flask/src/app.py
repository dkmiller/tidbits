from flask import Flask, jsonify, request
from src.ml import get_prediction

app = Flask(__name__)

@app.route('/')
def hello():
    return 'Hi there!'

@app.route('/predict', methods=['POST'])
def predict():
    image = request.files['file'].read()
    id, name = get_prediction(image)
    return jsonify({'class_id': id, 'class_name': name})
