import numpy as np
import os

# Error: AttributeError: module 'numpy' has no attribute 'object'.
# numpy > 1.23.4 defines np.object as object, etc.
# module 'numpy' has no attribute 'object'
np.object = object
np.bool = bool

# ImportError: dlopen(/Users/ren/anaconda3/envs/tensorflow/lib/python3.9/site-packages/array_record/python/array_record_module.so, 2): no suitable image found.  Did find:
#   /Users/ren/anaconda3/envs/tensorflow/lib/python3.9/site-packages/array_record/python/array_record_module.so: unknown file type, first eight bytes: 0x7F 0x45 0x4C 0x46 0x02 0x01 0x01 0x03
from tflite_model_maker.config import ExportFormat
from tflite_model_maker import model_spec
from tflite_model_maker import object_detector

import tensorflow as tf
assert tf.__version__.startswith('2')

tf.get_logger().setLevel('ERROR')
from absl import logging
logging.set_verbosity(logging.ERROR)

label_map = {1: 'player', 2: 'goalkeeper', 3:'referee', 4: 'ball'} 

train_images_dir = '/content/drive/MyDrive/Data Split/trainimage'
train_annotations_dir = '/content/drive/MyDrive/Data Split/trainanno'
val_images_dir = '/content/drive/MyDrive/Data Split/validimage'
val_annotations_dir = '/content/drive/MyDrive/Data Split/validanno'
test_images_dir = '/content/drive/MyDrive/Data Split/testimage'
test_annotations_dir = '/content/drive/MyDrive/Data Split/testanno'

train_data = object_detector.DataLoader.from_pascal_voc(
    train_images_dir, train_annotations_dir, label_map=label_map)

validation_data = object_detector.DataLoader.from_pascal_voc(
    val_images_dir, val_annotations_dir, label_map=label_map)

test_data = object_detector.DataLoader.from_pascal_voc(
    test_images_dir, test_annotations_dir, label_map=label_map)

spec = model_spec.get('efficientdet_lite0')

model = object_detector.create(train_data=train_data, 
                               model_spec=spec, 
                               validation_data=validation_data, 
                               epochs=50, 
                               batch_size=10, 
                               train_whole_model=True)

model.evaluate(test_data)

TFLITE_FILENAME = 'trained_model.tflite'
LABELS_FILENAME = 'model-labels.txt'

model.export(export_dir='.', tflite_filename=TFLITE_FILENAME, label_filename=LABELS_FILENAME,
             export_format=[ExportFormat.TFLITE, ExportFormat.LABEL])
