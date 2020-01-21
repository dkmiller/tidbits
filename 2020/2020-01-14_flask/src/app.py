import logging
from flask import Flask, jsonify, request
from src.ml import get_prediction

app = Flask(__name__)
# TODO: why isn't this working?
logging.basicConfig(level=logging.DEBUG)

@app.route('/')
def hello():
    return 'Hi there!'

@app.route('/predict', methods=['POST'])
def predict():
    # app.logger.error(f'Request: {request.__dict__}')
    # app.logger.error(f'Files: {request.files}')
    image = request.files['file'].read()
    id, name = get_prediction(image)
    return jsonify({'class_id': id, 'class_name': name})
