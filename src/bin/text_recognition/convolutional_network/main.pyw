import numpy as np
import PIL
from PIL import Image
import queue
import os.path

import keras
from keras.preprocessing.image import *
from keras.models import *
from keras.layers import *
from keras.utils import *
from keras.optimizers import *
from keras.callbacks import *
import os
import matplotlib.pyplot as plt


toId = {'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4, 'f': 5, 'g': 6, 'h': 7, 'i': 8, 'j': 9, 'k': 10, 'l': 11, 'm': 12, 'n': 13, 'o': 14, 'p': 15, 'q': 16, 'r': 17, 's': 18, 't': 19, 'u': 20, 'v': 21, 'w': 22, 'x': 23, 'y': 24, 'z': 25, 'A': 26, 'B': 27, 'C': 28, 'D': 29, 'E': 30, 'F': 31, 'G': 32, 'H': 33, 'I': 34, 'J': 35, 'K': 36, 'L': 37, 'M': 38, 'N': 39, 'O': 40, 'P': 41, 'Q': 42, 'R': 43, 'S': 44, 'T': 45, 'U': 46, 'V': 47, 'W': 48, 'X': 49, 'Y': 50, 'Z': 51, '0': 52, '1': 53, '2': 54, '3': 55, '4': 56, '5': 57, '6': 58, '7': 59, '8': 60, '9': 61, '_': 62}
toChar = {0: 'a', 1: 'b', 2: 'c', 3: 'd', 4: 'e', 5: 'f', 6: 'g', 7: 'h', 8: 'i', 9: 'j', 10: 'k', 11: 'l', 12: 'm', 13: 'n', 14: 'o', 15: 'p', 16: 'q', 17: 'r', 18: 's', 19: 't', 20: 'u', 21: 'v', 22: 'w', 23: 'x', 24: 'y', 25: 'z', 26: 'A', 27: 'B', 28: 'C', 29: 'D', 30: 'E', 31: 'F', 32: 'G', 33: 'H', 34: 'I', 35: 'J', 36: 'K', 37: 'L', 38: 'M', 39: 'N', 40: 'O', 41: 'P', 42: 'Q', 43: 'R', 44: 'S', 45: 'T', 46: 'U', 47: 'V', 48: 'W', 49: 'X', 50: 'Y', 51: 'Z', 52: '0', 53: '1', 54: '2', 55: '3', 56: '4', 57: '5', 58: '6', 59: '7', 60: '8', 61: '9', 62: '_'}


def isBlack(x):
    return x < 100

def process(dir="", name="img.jpg"):
    img = Image.open(os.path.join(dir, name))
    name = name.split(".")[0]
    img = img.convert("L")
    img = np.array(img)
    img = img * 255. / img.max()
    img[img > 220] = 255
    img[img < 255] = 0
    #img = np.array([list(map(lambda x: 0 if x < 220 else 255, img_row)) for img_row in img])
    img = PIL.Image.fromarray(np.uint8(img))
    w, h = img.size
    img = np.array(img)
    mask = np.zeros(img.shape)
    boxes = []

    for j in range(1, w-1):
        for i in range(1, h-1):
            s = True
            for a in range(-1, 2):
                for b in range(-1, 2):
                    if (a != 0 or b != 0) and img[i+a][j+b] == 0:
                        s = False
            if s:
                mask[i][j] = 1
                img[i][j] = 255

    for j in range(w):
        for i in range(h):
            if isBlack(img[i][j]) and mask[i][j] == 0:
                q = queue.Queue(maxsize=w*h)
                q.put((i, j))
                xmin, xmax, ymin, ymax = i, i, j, j
                while not q.empty():
                    x, y = q.get()
                    for a in range(-2, 2+1):
                        for b in range(-1, 1+1):
                            if x+a >= 0 and x+a < h and y+b >= 0 and y+b < w and mask[x+a][y+b] == 0 and isBlack(img[x+a][y+b]):
                                mask[x+a][y+b] = 1
                                q.put((x+a, y+b))
                                xmin = min(xmin, x+a)
                                xmax = max(xmax, x+a)
                                ymin = min(ymin, y+b)
                                ymax = max(ymax, y+b)
                dex = xmax - xmin
                dey = ymax - ymin
                if dex > 5 and dey > 5 and 1.3 * dex > dey:
                    boxes.append((xmin, xmax, ymin, ymax))

    n = len(boxes)
    """l = 0
    for box in boxes:
        xmin, xmax, ymin, ymax = box
        l += xmax
    l /= n"""

    imgBoxes = np.copy(img)

    print(n)
    for box in boxes:
        xmin, xmax, ymin, ymax = box
        for k in range(xmin, xmax+1):
            imgBoxes[k][ymin] = 192
            imgBoxes[k][ymax] = 192
        for k in range(ymin, ymax+1):
            imgBoxes[xmin][k] = 192
            imgBoxes[xmax][k] = 192

    #PIL.Image.fromarray(np.uint8(imgBoxes)).show()
    img = PIL.Image.fromarray(np.uint8(img))

    res = []

    for box in boxes:
        xmin, xmax, ymin, ymax = box
        res.append(np.array(img.crop((ymin-2, xmin-2, ymax+2, xmax+2)).resize((28,28))).reshape((28,28,1)))
        #s.save("out/%s_%s.jpg" % (name, i))

    return np.array(res)

'''
for d in data:
    plt.imshow(d)
    plt.show()
'''

model = load_model("../../../res/convolutional_network/trained_networks/model_0.9329398.model")
model.summary()

for dir in ["../../../res/datasets/words"]:  #predict all images
    for name in os.listdir(dir):
        data = process("../../../res/datasets/words", name)

        res = model.predict(data, verbose=1)

        res = np.argmax(res, axis=1)

        pred = "".join([toChar[x] for x in res])

        #show results (one at a time)
        plt.imshow(Image.open(os.path.join("../../../res/datasets/words", name)))
        plt.title(pred)
        plt.show()