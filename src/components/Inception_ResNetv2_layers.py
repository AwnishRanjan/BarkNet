from src.logger import logging
from src.exception import CustomException 
import os , sys

from keras.layers import Dense, Flatten , Dropout , BatchNormalization , Activation 
from keras.models import  Sequential 
from keras.models import Model
from keras.applications import InceptionResNetV2
from keras.callbacks import EarlyStopping, ModelCheckpoint, Callback
from keras import optimizers
from keras import layers 
import keras
# import tensorflow as tf 

from keras.optimizers import Adam

inceptionresnetv2 = InceptionResNetV2(include_top=False , input_shape=(128,128,3))

class ModelTrainerConfig:
    def __init__(self):
        self.weights_folder_path = 'weights'
        if not os.path.exists(self.weights_folder_path):
            os.makedirs(self.weights_folder_path)

class model_initialisation:
    def __init__(self):
        logging.info("model_initialisation starts from layers file ")
        pass

    def build_inception_resnet_model(print_summary=True, 
                                input_shape=(128,128,3), num_classes=121):
        base_model = InceptionResNetV2(include_top=False, input_shape=input_shape, weights='imagenet')

        for layer in base_model.layers:
            layer.trainable = False
        x = base_model.output

        x = Flatten()(x)
        x = Dense(units=num_classes, activation='softmax')(x)

        model = keras.Model(inputs=base_model.input, outputs=x, name='Inception_ResNetv2')

        if print_summary:
            model.summary()

        return model


    
class Model:
    def __init__(self):
        self.config = ModelTrainerConfig()

    def build_model(self):
        try:
            earlystopping = EarlyStopping(monitor='val_accuracy', patience=5)
            checkpoint_filepath = '/Users/awnishranjan/Developer/currentproject/weights/model.weights.h5'
            model_checkpoint_callback = ModelCheckpoint(
                filepath=checkpoint_filepath,
                save_weights_only=True,
                monitor='val_accuracy',
                verbose=2,
                mode='max',
                save_best_only=True)

            # Build the model
            model_initialiser = model_initialisation()
            model = model_initialiser.build_inception_resnet_model()
            optimizer = optimizers.Adam()

            # Compile
            model.compile(loss='categorical_crossentropy',
                            optimizer=optimizer,
                            metrics=['accuracy'])
            
            logging.info("model building completed ")
            
            return earlystopping,model_checkpoint_callback,model
        
            
        
        except Exception as e:
            logging.error(f"Error building model from layers : {e}")
            raise CustomException("Error building model ", e)




            