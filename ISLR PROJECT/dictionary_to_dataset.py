import os
import numpy as np
import random
import cv2


def create_dataset(data_dir, categories, resize_shape):
    dataset = []
    features = []
    labels = []
    for category in categories:


        path = os.path.join(data_dir,category)
        class_num = categories.index(category)

        for count, img in enumerate(os.listdir(path)):
            original_img = cv2.imread(os.path.join(path,img))
            gray_img = cv2.cvtColor(original_img, cv2.COLOR_RGB2GRAY)
            gray_img = cv2.resize(gray_img, resize_shape)
            dataset.append([gray_img, class_num])

    #suffle the dataset 
    random.shuffle(dataset)

    #append all features and labels from dataset to saprate list
    for feature, label in dataset:
        features.append(feature)
        labels.append(label)

    #convert features and labels to numpy array       
    features = np.array(features)
    labels = np.array(labels)

    return (features, labels)