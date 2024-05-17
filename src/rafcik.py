# from PIL import Image
# from keras.src.utils import load_img
# import numpy as np
# import pandas as pd
# import tensorflow as tf
# import matplotlib.pyplot as plt
# import warnings
# import random
# import os
import cv2
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

def extract_color_histogram(image, bins=(8, 8, 8)):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, bins, [0, 180, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()


# Load data dari folder
def load_data(data_path):
    data = []
    labels = []
    for label in ["matang", "belum_matang"]:
        dir_path = os.path.join(data_path, label)
        for file_name in os.listdir(dir_path):
            file_path = os.path.join(dir_path, file_name)
            image = cv2.imread(file_path)
            if image is not None:
                features = extract_color_histogram(image)
                data.append(features)
                labels.append(label)
    return np.array(data), np.array(labels)

# Path ke folder data
data_path = '../dataset/'

# Load data dan label
data, labels = load_data(data_path)
print(labels)