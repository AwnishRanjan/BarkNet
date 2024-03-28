import sys 
import os 
import pandas as pd
import numpy as np
from src.logger import logging
from src.exception import CustomException
# import shutil
from sklearn.model_selection import train_test_split

class DataInjectionConfig:
    train_images_path: str = os.path.join('artifacts', 'train_images')
    test_images_path: str = os.path.join('artifacts', 'test_images')
    train_annotations_path: str = os.path.join('artifacts', 'train_annotations')
    test_annotations_path: str = os.path.join('artifacts', 'test_annotations')
    # raw_images_path: str = os.path.join('artifacts', 'raw_images')
    # raw_annotations_path: str = os.path.join('artifacts', 'raw_annotations')

# class data_injection:
#     def __init__(self):
#         self.injection_config = DataInjectionConfig()

#     def initiate_data_injection(self, image_dir, annotation_dir):
#         logging.info('Data injection method started')

#         try:
#             os.makedirs(self.injection_config.train_images_path, exist_ok=True)
#             os.makedirs(self.injection_config.test_images_path, exist_ok=True)
#             os.makedirs(self.injection_config.train_annotations_path, exist_ok=True)
#             os.makedirs(self.injection_config.test_annotations_path, exist_ok=True)

#             image_files = [f for f in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, f))]
#             # annotation_files = [f for f in os.listdir(annotation_dir) if os.path.isfile(os.path.join(annotation_dir, f))]
#             # print(len(annotation_files))
#             train_files, test_files = train_test_split(image_files, test_size=0.1, random_state=42)
#             # train_annotation_files, test_annotation_files = train_test_split(annotation_files, test_size=0.1, random_state=42)

#             # train_annotation_csv = os.path.join(self.injection_config.raw_annotations_path, 'train_annotations.csv')
#             # test_annotation_csv = os.path.join(self.injection_config.raw_annotations_path, 'test_annotations.csv')
#             # pd.DataFrame({'file_name': train_annotation_files}).to_csv(train_annotation_csv, index=False)
#             # pd.DataFrame({'file_name': test_annotation_files}).to_csv(test_annotation_csv, index=False)

#             for file in train_files:
#                 src_img = os.path.join(image_dir, file)
#                 dst_img = os.path.join(self.injection_config.train_images_path, file)
#                 self.copy_file(src_img, dst_img)
#                 src_ann = os.path.join(annotation_dir, file.replace('.jpg', '.xml'))
#                 dst_ann = os.path.join(self.injection_config.train_annotations_path, file.replace('.jpg', '.xml'))
#                 self.copy_file(src_ann, dst_ann)

#             for file in test_files:
#                 src_img = os.path.join(image_dir, file)
#                 dst_img = os.path.join(self.injection_config.test_images_path, file)
#                 self.copy_file(src_img, dst_img)
#                 src_ann = os.path.join(annotation_dir, file.replace('.jpg', '.xml'))
#                 dst_ann = os.path.join(self.injection_config.test_annotations_path, file.replace('.jpg', '.xml'))
#                 self.copy_file(src_ann, dst_ann)
            
#             logging.info('Data injection completed')
#             return (
#                 self.injection_config.train_images_path,
#                 self.injection_config.train_annotations_path,
#                 self.injection_config.test_images_path,
#                 self.injection_config.test_annotations_path
#             )

#         except Exception as e:
#             logging.info('Error occurred in data injection:')
#             raise CustomException(e,sys)

#     def copy_file(self, src, dst):
#         if os.path.exists(src):
#             with open(src, 'rb') as fsrc, open(dst, 'wb') as fdst:
#                 fdst.write(fsrc.read())

class data_injection:
    def __init__(self):
        self.injection_config = DataInjectionConfig()

    def initiate_data_injection(self, image_dir, annotation_dir):
        logging.info('Data injection method started')

        try:
            os.makedirs(self.injection_config.train_images_path, exist_ok=True)
            os.makedirs(self.injection_config.test_images_path, exist_ok=True)
            
            # os.makedirs(self.injection_config.train_annotations_path, exist_ok=True)
            # os.makedirs(self.injection_config.test_annotations_path, exist_ok=True)
            # os.makedirs(self.injection_config.raw_annotations_path, exist_ok=True)


            # Get list of breed folders
            breed_folders = [f for f in os.listdir(image_dir) if os.path.isdir(os.path.join(image_dir, f))]

            for breed_folder in breed_folders:
                breed_images_dir = os.path.join(image_dir, breed_folder)
                breed_annotations_dir = os.path.join(annotation_dir, breed_folder)

                train_images, test_images = train_test_split(os.listdir(breed_images_dir), test_size=0.1, random_state=42)
                train_annotations, test_annotations = train_test_split(os.listdir(breed_annotations_dir), test_size=0.1, random_state=42)

                    
                # train_annotation_csv = os.path.join(self.injection_config.raw_annotations_path, 'train_annotations.csv')
                # test_annotation_csv = os.path.join(self.injection_config.raw_annotations_path, 'test_annotations.csv')
                # pd.DataFrame({'file_name': train_annotations}).to_csv(train_annotation_csv, index=False)
                # pd.DataFrame({'file_name': test_annotations}).to_csv(test_annotation_csv, index=False)

                # train data 
                for image_file in train_images:
                    src_img = os.path.join(breed_images_dir, image_file)
                    dst_img = os.path.join(self.injection_config.train_images_path, breed_folder, image_file)
                    self.copy_file(src_img, dst_img)

                    annotation_file = image_file.replace('.jpg', '.xml')  
                    src_ann = os.path.join(breed_annotations_dir, annotation_file)
                    dst_ann = os.path.join(self.injection_config.train_annotations_path, breed_folder, annotation_file)
                    self.copy_file(src_ann, dst_ann)
                # test data
                for image_file in test_images:
                    src_img = os.path.join(breed_images_dir, image_file)
                    dst_img = os.path.join(self.injection_config.test_images_path, breed_folder, image_file)
                    self.copy_file(src_img, dst_img)

                    annotation_file = image_file.replace('.jpg', '.xml')  
                    src_ann = os.path.join(breed_annotations_dir, annotation_file)
                    dst_ann = os.path.join(self.injection_config.test_annotations_path, breed_folder, annotation_file)
                    self.copy_file(src_ann, dst_ann)
            
            logging.info('Data injection completed')
            return (
                self.injection_config.train_images_path,
                self.injection_config.test_images_path
            )

        except Exception as e:
            logging.info('Error occurred in data injection:')
            raise CustomException('Data injection failed', e)



    def copy_file(self, src, dst):
        if os.path.exists(src):
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            with open(src, 'rb') as fsrc, open(dst, 'wb') as fdst:
                fdst.write(fsrc.read())
