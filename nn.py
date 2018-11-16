import numpy as np
import os
import random
import cv2
from imutils import paths
import matplotlib.pyplot as plt


image_paths = list(paths.list_images('datasets/animals'))

random.seed(42)
random.shuffle(image_paths)

image = cv2.imread(image_paths[2500])
plt.figure(figsize=(10, 10))
rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
plt.imshow(rgb_image)
plt.savefig("gif.jpg")


data = []
labels = []

for imagePath in image_paths:
    image = cv2.imread(imagePath)
    label = imagePath.split(os.path.sep)[-2]
    image = cv2.resize(image, (32, 32), interpolation=cv2.INTER_AREA)
    data.append(image)
    labels.append(label)


data = np.array(data)
labels = np.array(labels)

plt.figure(figsize=(10, 10))
rgb_image = cv2.cvtColor(data[2500], cv2.COLOR_BGR2RGB)
plt.imshow(rgb_image)
plt.savefig("resize_gif.jpg")


from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

data = data.reshape((data.shape[0], 3072))
data = data.astype('float') /  255.0


le = LabelEncoder()
labels = le.fit_transform(labels)

(trainX, testX, trainY,testY) = train_test_split(data, labels, test_size= 0.25, random_state= 30)

nn = MLPClassifier()
nn.fit(trainX,trainY)

y_pred= nn.predict(testX)

print(classification_report(testY, y_pred, target_names=le.classes_))