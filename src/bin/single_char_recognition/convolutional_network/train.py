import os
import os.path
import numpy as np
import pandas as pd
import tensorflow.keras
from tensorflow.keras.preprocessing.image import *
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.utils import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.callbacks import *
from skimage.io import imread
import matplotlib.pyplot as plt
import gc; gc.enable()
from cv2 import *

# Settings
showImages = False
mnistDir = "../../../res/datasets/emnist_balanced"

# Load eMNIST
toId = {}
toChar = {}
with open(os.path.join(mnistDir, "emnist_balanced_mapping_to_char.txt")) as f:
    for line in f.read().split("\n"):
        a, b = line.split(" ")
        toChar[int(a)] = b
        toId[b] = a
numClasses = len(toChar)

# Debug
#print(numClasses, "classes")
#print("toChar =", toChar)
#print("toId =", toId)

train = pd.read_csv(os.path.join(mnistDir, "emnist_balanced_train_uppercase.csv"), header=None).to_numpy()
x = np.transpose(train[:, 1:].reshape((-1, 28, 28, 1)), (0, 2, 1, 3))
y = to_categorical(train[:, 0], num_classes=numClasses)
x = (x - x.mean()) / x.std()

# Debug
#print(x.shape)
#print(y.shape)

generator = ImageDataGenerator(
    rotation_range=15,
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=False,
    vertical_flip=False
)
generator.fit(x)

if showImages:
    for i in range(100):
        d, l = x[i], y[i]
        print(toChar[np.argmax(l)])
        plt.title("" + str(toChar[np.argmax(l)]) + " - " + str(l))
        plt.imshow(d[:, :, 0], cmap="Greys")
        plt.show()

lrDecay = LearningRateScheduler(lambda epoch: 0.001 * np.power(0.95, epoch))
#saver = ModelCheckpoint("model1.model")

model = Sequential()
model.add(InputLayer(input_shape=(28, 28, 1)))
model.add(Conv2D(32, kernel_size=3, activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(32, kernel_size=3, activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(32, kernel_size=5, strides=2, padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.4))
model.add(Conv2D(64, kernel_size=3, activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(64, kernel_size=3, activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(64, kernel_size=5, strides=2, padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.4))
model.add(Conv2D(128, kernel_size=4, activation='relu'))
model.add(BatchNormalization())
model.add(Flatten())
model.add(Dropout(0.4))
model.add(Dense(numClasses, activation='softmax'))

model.compile(optimizer=Adam(0.001), loss="categorical_crossentropy", metrics=["acc"])
model.summary()

stepsPerEpoch = 128
model.fit_generator(generator.flow(x, y, batch_size=stepsPerEpoch), steps_per_epoch=len(x) // stepsPerEpoch, epochs=1, callbacks=[lrDecay, saver])

loss, acc = model.evaluate(x, y)
model.save("../../../res/convolutional_network/trained_networks/model_%s.model" % (acc))