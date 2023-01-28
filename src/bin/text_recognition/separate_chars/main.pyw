import numpy as np
import PIL
from PIL import Image
import queue
import os.path

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

    PIL.Image.fromarray(np.uint8(imgBoxes)).show()
    img = PIL.Image.fromarray(np.uint8(img))


    if len(boxes) == len(name):
        i = 0
        for box in boxes:
            xmin, xmax, ymin, ymax = box
            s = img.crop((ymin-2, xmin-2, ymax+2, xmax+2))
            s.save("out/%s_%s.jpg" % (name, i))
            i += 1
    else:
        log.write(name + "\n")


#log = open("log.txt", "w")
#
#for dir in ["all"]:
#    for name in os.listdir(dir):
#        print(dir, name)
#        process(dir, name)
#
#log.close()

process()