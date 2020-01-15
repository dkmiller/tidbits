from flask import Flask, jsonify

app = Flask(__name__)

@app.route('/')
def hello():
    return 'Hi there!'

@app.route('/predict', methods=['POST'])
def predict():
    return jsonify({'class_id': 'IMAGE_NET_XX', 'class_name': 'Cat'})
