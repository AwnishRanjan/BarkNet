from src.components.model_prediction import prediction
from src.components.model_prediction import decode

from src.logger import logging
# from src.exception import CustomException 
import numpy as np 



if __name__ == '__main__':
    logging.info("prediction pipeline initiated ")
    initial_model = prediction()  

    model = initial_model.load_model('artifacts/models/trained_model.keras')

    logging.log("model initilisation completed ")


    img_path = 'artifacts/test_images/n02106662-German_shepherd/n02106662_1094.jpg'
    preprocessed_image = initial_model.preprocess_image(img_path)

    if preprocessed_image is not None:
        predictions = model.predict(np.expand_dims(preprocessed_image, axis=0))
     
    decoder = decode()
    decoder.load_index_to_label()
    output = decoder.decode_prediction(predictions)

    logging.info(f'output for upload image is  {output}')

    print(output)



    





