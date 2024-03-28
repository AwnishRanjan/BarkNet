import numpy as np
import os 
import json
from src.logger import logging
from src.exception import CustomException 
import cv2 
from keras.utils import to_categorical


class DataTransformation:
    def __init__(self):
        pass 

    def load_images_from_folder(self,folder, target_size):
        images = []
        labels = []
        for breed_folder in os.listdir(folder):
            breed_path = os.path.join(folder, breed_folder)
            if os.path.isdir(breed_path):
                for filename in os.listdir(breed_path):
                    img_path = os.path.join(breed_path, filename)
                    img = cv2.imread(img_path)
                    if img is not None:
                        img_resized = cv2.resize(img, target_size)  # Resize image
                        images.append(img_resized)
                        labels.append(breed_folder)
        logging.info("load_image completed ")
        return images, labels
   
    def preprocess_data(self, images, labels, num_classes):
        try:
            np.random.seed(42)
            # Create label-to-index mapping
            label2index = dict((name, index) for index, name in enumerate(np.unique(labels)))
            index2label = dict((index, name) for name, index in label2index.items())
            processed_y = np.asarray([label2index[label] for label in labels])
            processed_y = to_categorical(processed_y, num_classes=num_classes)
            images = np.array(images)

            # Save index-to-label mapping to a JSON file in the artifacts folder
            artifacts_dir = 'artifacts'
            os.makedirs(artifacts_dir, exist_ok=True)
            index2label_file = os.path.join(artifacts_dir, 'index_to_label_mapping.json')
            with open(index2label_file, 'w') as f:
                json.dump(index2label, f)

            logging.info("preprocess_data completed")
        except Exception as e:
            logging.error(f"Error preprocessing data: {e}")
            raise CustomException("Error preprocessing data", e)

        return images, processed_y
    


    
