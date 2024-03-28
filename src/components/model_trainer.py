import os, sys 
from src.logger import logging 
from src.exception import CustomException
import tensorflow as tf
from src.components.Inception_ResNetv2_layers import Model
import time

# required to load images from paths but we dont need it 
# def load_image_partial(path, label, num_channels, image_height, image_width):
#     print("Path:", path)  # Add this line to print the value of path
#     image = tf.io.read_file(path)
#     image = tf.image.decode_jpeg(image, channels=num_channels)
#     image = tf.image.resize(image, [image_height, image_width])
#     return image, label

def normalize(image, label):
        image = image / 255
        return image, label

class ModelTraining:

    def __init__(self):
        pass
    
    def build_data_generators(self, X_train, y_train, X_valid, y_valid, X_test, y_test,
                           train_batch_size, validation_batch_size, test_batch_size,epochs,
                           train_data_process_list=[normalize],
                           validate_data_process_list=[normalize],
                           test_data_process_list=[normalize]):
        try:
            logging.info("build_data_generator starts here ...")
            train_data_count = len(X_train)
            validation_data_count = len(X_valid)
            test_data_count = len(X_test)
            logging.info("step 0 completed")

            train_data = tf.data.Dataset.from_tensor_slices((X_train, y_train))
            validation_data = tf.data.Dataset.from_tensor_slices((X_valid, y_valid))
            test_data = tf.data.Dataset.from_tensor_slices((X_test, y_test))
            logging.info("step 1 completed")

            AUTOTUNE = tf.data.experimental.AUTOTUNE

            train_data = train_data.shuffle(len(X_train))
            for process in train_data_process_list:
                train_data = train_data.map(process, num_parallel_calls=AUTOTUNE)
            train_data = train_data.repeat(epochs).batch(train_batch_size).prefetch(AUTOTUNE)

            logging.info("step 2 completed")

            validation_data = validation_data.shuffle(validation_data_count)
            for process in validate_data_process_list:
                validation_data = validation_data.map(process, num_parallel_calls=AUTOTUNE)
            validation_data = validation_data.repeat(epochs).batch(validation_batch_size).prefetch(AUTOTUNE)
            
            logging.info("step 3 completed")

            for process in test_data_process_list:
                test_data = test_data.map(process, num_parallel_calls=AUTOTUNE)

            test_data = test_data.repeat(1).batch(test_batch_size).prefetch(AUTOTUNE)

            
            logging.info("image data encoded successfully ")

            return train_data, validation_data, test_data

        except Exception as e:
            logging.error(f"Error building data generators: {e}")
            raise CustomException("Error building data generators", e)
        





    def train_model(self, train_data, valid_data, batch_size, epochs):
        model_instance = Model()

        earlystopping, model_checkpoint_callback, model = model_instance.build_model()

        try:
            steps_per_epoch = train_data.cardinality().numpy() // batch_size
            
            # Start training
            start_time = time.time()
            history = model.fit(
                train_data,
                validation_data=valid_data,
                initial_epoch=10,
                epochs=epochs,
                callbacks=[earlystopping, model_checkpoint_callback],
                verbose=1,
                steps_per_epoch=steps_per_epoch)
            
            execution_time = (time.time() - start_time) / 60.0
            print("Training execution time (mins)", execution_time)

             # Save the model
            model_dir = os.path.join('artifacts', 'models')
            os.makedirs(model_dir, exist_ok=True)
            model_path = os.path.join(model_dir, 'trained_model.keras')
            model.save(model_path)

            logging.info("Model saved at:", model_path)
            
        except Exception as e:
            logging.error(f"An error occurred during training: {e}")
            raise CustomException("Error during training", e)
            
        return history


