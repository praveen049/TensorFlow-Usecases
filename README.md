# TensorFlow-Usecases
Sample implementations for TensorFlow-Usecases

Contains a bunch of Neural Networks code examples using Tensorflow.

Includes exmaples using different wrappers including tflayer and keras.



CLASSIFICATION EXAMPLES

Dataset:
  A) prt dataset is a labelled dataset mapping a set of logs file to a group. 
    X = text content
    y = a group name

Preprocessing:
  The dataset is read using pandas and title and description fields are combined to create the input X
  The group name is mapped to y
  
Example:
  multi_class_bot1.py
  This examples does multi class classificatin with keras API with the prt dataset.
  
