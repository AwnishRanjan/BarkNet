from flask import Flask, request, jsonify, render_template
from src.components.model_prediction import prediction, decode
import numpy as np
from src.logger import logging

app = Flask(__name__)
initiate_model = prediction()
model = initiate_model.load_model('artifacts/models/trained_model.keras')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        logging.info("app step 1 completed ")
        image_file = request.files['image']
        image_path = 'temp_image.jpg'
        image_file.save(image_path)
        logging.info("app.py step 2 completed ")

        preprocessed_image = initiate_model.preprocess_image(image_path)
        logging.info("app.py step 3 completed ")

        if preprocessed_image is not None:
            predictions = model.predict(np.expand_dims(preprocessed_image, axis=0))
            decoder = decode()
            decoder.load_index_to_label()
            decoded_predictions = decoder.decode_prediction(predictions)

            return jsonify({'predictions': decoded_predictions})
        else:
            return jsonify({'error': 'Failed to preprocess the image'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000)
