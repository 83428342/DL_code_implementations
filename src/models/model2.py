import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from data.mnist import load_mnist

(x_train, t_train), (x_test, t_test) = load_mnist()

print(x_train.shape)
print(t_train.shape)
print(x_test.shape)
print(t_test.shape)

import sys, os
import numpy as np
from PIL import Image

def img_show(img):
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()

img = x_train[0]
label = t_train[0]
print(label)

print(img.shape)
img = img.reshape(28, 28)
print(img.shape)

img_show(img)
