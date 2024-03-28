from src.logger import logging
# from src.exception import CustomException
from src.components.data_injection import data_injection
from src.components.data_transformation import DataTransformation

from sklearn.model_selection import train_test_split
import os ,json
from src.components.model_trainer import ModelTraining
import numpy as np
import tensorflow as tf
from src.utils import evaluate_save_model
if __name__ == '__main__':
    obj = data_injection()

    img_dir = '/Users/awnishranjan/Developer/currentproject/notebook/data/Images'
    annotation_dir = '/Users/awnishranjan/Developer/currentproject/notebook/data/Annotation'
  

    train_img_path ,  test_img_path = obj.initiate_data_injection(img_dir,annotation_dir)

    print(train_img_path , test_img_path )

    data_transform = DataTransformation()
    
    target_size= (128,128)

    train_images, train_labels = data_transform.load_images_from_folder(train_img_path,target_size)
    test_images , test_labels = data_transform.load_images_from_folder(test_img_path,target_size)
    X_train, y_train = data_transform.preprocess_data(train_images, train_labels,num_classes=121)
    X_test, y_test = data_transform.preprocess_data(test_images, test_labels,num_classes=121)

    X_train,X_valid,y_train,y_valid= train_test_split(X_train,y_train, test_size=0.2,random_state=42) 

    logging.info(f' test length is {len(y_test)}')
    logging.info(f' valid length is {len(y_valid)}')
    logging.info(f' train length is {len(y_train)}')

    logging.info("X_train type: %s", type(X_train))
    logging.info("y_train type: %s", type(y_train))
    logging.info("X_valid type: %s", type(X_valid))
    logging.info("y_valid type: %s", type(y_valid))
    logging.info("X_test type: %s", type(X_test))
    logging.info("y_test type: %s", type(y_test))

    if isinstance(X_train, np.ndarray):
        logging.info("X_train shape: %s", X_train.shape)
    if isinstance(y_train, np.ndarray):
        logging.info("y_train shape: %s", y_train.shape)
    if isinstance(X_valid, np.ndarray):
        logging.info("X_valid shape: %s", X_valid.shape)
    if isinstance(y_valid, np.ndarray):
        logging.info("y_valid shape: %s", y_valid.shape)
    if isinstance(X_test, np.ndarray):
        logging.info("X_test shape: %s", X_test.shape)
    if isinstance(y_test, np.ndarray):
        logging.info("y_test shape: %s", y_test.shape)

    validation_percent = 0.2
    image_width = 128
    image_height = 128
    num_channels = 3
    num_classes = 121
    epochs = 2
    train_batch_size = 32
    validation_batch_size = 32
    test_batch_size = 32

    model_training = ModelTraining()

    train_data, validation_data, test_data = model_training.build_data_generators(X_train, y_train, X_valid, y_valid, X_test, y_test,
                                             train_batch_size, validation_batch_size, test_batch_size,epochs)
    

    history = model_training.train_model(train_data,validation_data , train_batch_size, epochs=11)
    history_file_path = os.path.join('artifacts', 'training_history.json')
    with open(history_file_path, 'w') as f:
            json.dump(history.history, f)

    trained_model = tf.keras.models.load_model('/Users/awnishranjan/Developer/currentproject/artifacts/models/trained_model.keras')

    # test_loss, test_accuracy = trained_model.evaluate(X_test, y_test)
    # logging.info(f'Test Loss: {test_loss}')
    # logging.info(f'Test Accuracy: {test_accuracy}')

    directory = 'images'

    # evaluate_save_model(history,directory)

