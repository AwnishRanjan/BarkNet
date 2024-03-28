from src.logger import logging
from src.exception import CustomException 
import json 
from keras.models import load_model
import cv2
import numpy as np 
import os 


def resize_image(image, size):
    resized_image = cv2.resize(image, size)
    return resized_image

def normalize_image(image):
    normalized_image = image.astype(np.float32) / 255.0
    return normalized_image
    

class prediction:
    def __init__(self):
        pass

    def load_model(self,path):
        model = load_model(path)
        return model
 
    def preprocess_image(self, image_path, target_size=(128, 128)):
        image = cv2.imread(image_path)
        if image is None:
            logging.error("Failed to load the image at path: %s", image_path)
            return None
        resized_image = resize_image(image, target_size)
        normalized_image = normalize_image(resized_image)
        
        return normalized_image
    
class decode:
    def __init__(self):
        self.index_to_label = None

    def load_index_to_label(self, path='artifacts/index_to_label_mapping.json'):
        if not os.path.exists(path):
            logging.error("Index to label mapping file not found at path: %s", path)
            return
        with open(path, 'r') as f:
            self.index_to_label = json.load(f)

    def decode_prediction(self, prediction):
        if self.index_to_label is None:
            logging.error("Index to label mapping is not loaded.")
            return None
        predicted_index = np.argmax(prediction)
        predicted_label = self.index_to_label.get(str(predicted_index))

        return predicted_label


        
